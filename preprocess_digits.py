import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def foreground_bbox(image: np.ndarray, threshold: int) -> tuple[int, int, int, int] | None:
    mask = image < threshold
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1


def estimate_slant_deg(image: np.ndarray, threshold: int) -> float:
    mask = image < threshold
    if mask.sum() < 2:
        return 0.0

    y, x = np.nonzero(mask)
    weights = (255.0 - image[y, x].astype(np.float32)) / 255.0
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return 0.0

    x_mean = float((x * weights).sum() / weight_sum)
    y_mean = float((y * weights).sum() / weight_sum)
    x_centered = x - x_mean
    y_centered = y - y_mean
    var_y = float((weights * y_centered * y_centered).sum() / weight_sum)
    if var_y <= 1e-6:
        return 0.0

    cov_xy = float((weights * x_centered * y_centered).sum() / weight_sum)
    slope = cov_xy / var_y
    return float(np.degrees(np.arctan(slope)))


def preprocess_one(
    image: np.ndarray,
    output_size: int,
    content_size: int,
    threshold: int,
) -> tuple[np.ndarray, dict[str, float]]:
    bbox = foreground_bbox(image, threshold)
    if bbox is None:
        canvas = np.full((output_size, output_size), 255, dtype=np.uint8)
        return canvas, {
            "top": 0,
            "left": 0,
            "bottom": image.shape[0],
            "right": image.shape[1],
            "resized_h": output_size,
            "resized_w": output_size,
            "pad_top": 0,
            "pad_left": 0,
            "ink_ratio": 0.0,
            "thickness": 0.0,
            "slant_deg": 0.0,
        }

    top, left, bottom, right = bbox
    crop = image[top:bottom, left:right]
    crop_h, crop_w = crop.shape
    scale = content_size / max(crop_h, crop_w)
    resized_h = max(1, int(round(crop_h * scale)))
    resized_w = max(1, int(round(crop_w * scale)))

    pil_crop = Image.fromarray(crop)
    resized = pil_crop.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    resized_array = np.asarray(resized, dtype=np.uint8)

    canvas = np.full((output_size, output_size), 255, dtype=np.uint8)
    pad_top = (output_size - resized_h) // 2
    pad_left = (output_size - resized_w) // 2
    canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized_array

    final_mask = canvas < threshold
    ink_pixels = int(final_mask.sum())
    ink_ratio = ink_pixels / float(output_size * output_size)

    final_bbox = foreground_bbox(canvas, threshold)
    if final_bbox is None:
        thickness = 0.0
    else:
        f_top, f_left, f_bottom, f_right = final_bbox
        box_h = max(1, f_bottom - f_top)
        box_w = max(1, f_right - f_left)
        thickness = ink_pixels / float(box_h + box_w)

    return canvas, {
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "resized_h": resized_h,
        "resized_w": resized_w,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "ink_ratio": ink_ratio,
        "thickness": thickness,
        "slant_deg": estimate_slant_deg(canvas, threshold),
    }


def normalize_feature(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - lo) / (hi - lo)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop handwritten digits, resize with aspect ratio, and pad to 128x128."
    )
    parser.add_argument("--images", default="data/Images.npy")
    parser.add_argument("--writer-info", default="data/WriterInfo.npy")
    parser.add_argument("--out-images", default="data/Images_128.npy")
    parser.add_argument("--out-writer-info", default="data/WriterInfo_128.npy")
    parser.add_argument("--out-meta", default="data/PreprocessInfo_128.npz")
    parser.add_argument("--output-size", type=int, default=128)
    parser.add_argument("--content-size", type=int, default=112)
    parser.add_argument("--threshold", type=int, default=245)
    parser.add_argument("--preview-dir", default="images_128_preview")
    parser.add_argument("--preview-step", type=int, default=400)
    args = parser.parse_args()

    images_path = Path(args.images)
    writer_path = Path(args.writer_info)
    out_images_path = Path(args.out_images)
    out_writer_path = Path(args.out_writer_info)
    out_meta_path = Path(args.out_meta)
    preview_dir = Path(args.preview_dir)

    images = np.load(images_path, mmap_mode="r")
    writer_info = np.load(writer_path)
    if images.ndim != 3:
        raise ValueError(f"Expected images with shape (N, H, W), got {images.shape}")
    if writer_info.shape[0] != images.shape[0]:
        raise ValueError(
            f"WriterInfo rows ({writer_info.shape[0]}) do not match images ({images.shape[0]})"
        )

    out_images_path.parent.mkdir(parents=True, exist_ok=True)
    out_writer_path.parent.mkdir(parents=True, exist_ok=True)
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    output = np.lib.format.open_memmap(
        out_images_path,
        mode="w+",
        dtype=np.uint8,
        shape=(images.shape[0], args.output_size, args.output_size),
    )

    meta = np.zeros((images.shape[0], 13), dtype=np.float32)
    meta_columns = np.array(
        [
            "top",
            "left",
            "bottom",
            "right",
            "resized_h",
            "resized_w",
            "pad_top",
            "pad_left",
            "ink_ratio",
            "thickness",
            "slant_deg",
            "thickness_norm",
            "slant_norm",
        ]
    )

    for index in range(images.shape[0]):
        processed, info = preprocess_one(
            images[index],
            output_size=args.output_size,
            content_size=args.content_size,
            threshold=args.threshold,
        )
        output[index] = processed
        meta[index, :11] = [
            info["top"],
            info["left"],
            info["bottom"],
            info["right"],
            info["resized_h"],
            info["resized_w"],
            info["pad_top"],
            info["pad_left"],
            info["ink_ratio"],
            info["thickness"],
            info["slant_deg"],
        ]

        if args.preview_step > 0 and index % args.preview_step == 0:
            Image.fromarray(processed).save(preview_dir / f"{index}.png")

        if (index + 1) % 1000 == 0 or index + 1 == images.shape[0]:
            print(f"processed {index + 1}/{images.shape[0]}")

    output.flush()
    meta[:, 11] = normalize_feature(meta[:, 9])
    meta[:, 12] = normalize_feature(meta[:, 10])

    np.save(out_writer_path, writer_info)
    np.savez_compressed(
        out_meta_path,
        columns=meta_columns,
        values=meta,
        writer_info=writer_info,
        source_images=str(images_path),
        threshold=args.threshold,
        output_size=args.output_size,
        content_size=args.content_size,
    )

    print(f"saved images: {out_images_path} {tuple(output.shape)} {output.dtype}")
    print(f"saved writer info: {out_writer_path} {writer_info.shape} {writer_info.dtype}")
    print(f"saved metadata: {out_meta_path} columns={meta_columns.tolist()}")


if __name__ == "__main__":
    main()

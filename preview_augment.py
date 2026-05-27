import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from train import adjust_thickness, shear_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview training-time thickness and slant augmentations.")
    parser.add_argument("--images", default="data/Images_256.npy")
    parser.add_argument("--writer-info", default="data/WriterInfo_256.npy")
    parser.add_argument("--digit", type=int, default=3)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", default="cgan-images-256-clean/augment_preview.png")
    parser.add_argument("--threshold", type=int, default=210)
    parser.add_argument("--thickness-steps", type=int, default=4)
    parser.add_argument("--slant-strength", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = np.load(args.images, mmap_mode="r")
    writer_info = np.load(args.writer_info)
    matches = np.where(writer_info[:, 0] == args.digit)[0]
    if len(matches) == 0:
        raise ValueError(f"No samples for digit {args.digit}")
    sample_index = int(matches[min(args.index, len(matches) - 1)])
    base = np.where(images[sample_index] < args.threshold, 0, 255).astype(np.uint8)

    thickness_values = [0.05, 0.25, 0.5, 0.75, 0.95]
    slant_values = [0.05, 0.25, 0.5, 0.75, 0.95]
    rows = [
        ("thickness", [adjust_thickness(base, v, args.thickness_steps) for v in thickness_values], thickness_values),
        ("slant", [shear_image(base, v, args.slant_strength) for v in slant_values], slant_values),
    ]

    cell = 256
    label_w = 90
    header_h = 24
    row_h = 286
    canvas = Image.new("L", (label_w + len(thickness_values) * cell, header_h + len(rows) * row_h), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), f"source index {sample_index}", fill=0)

    for row, (name, row_images, values) in enumerate(rows):
        y = header_h + row * row_h
        draw.text((4, y + 8), name, fill=0)
        for col, image in enumerate(row_images):
            x = label_w + col * cell
            canvas.paste(Image.fromarray(image), (x, y))
            draw.text((x + 4, y + cell + 4), f"{values[col]:.2f}", fill=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"saved augment preview to {out_path}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from cgan_model import ConditionalGenerator


def parse_float_list(value: str, count: int, default: float) -> list[float]:
    if not value:
        return [default] * count
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"Expected one value or {count} comma-separated values, got {len(values)}.")
    return values


def save_grid(images: torch.Tensor, path: Path, nrow: int, clean: bool, threshold: int) -> None:
    images = images.detach().cpu().clamp(-1, 1)
    images = ((images + 1.0) * 127.5).to(torch.uint8)
    images = images.squeeze(1).numpy()

    count, height, width = images.shape
    nrow = min(nrow, count)
    ncol = int(np.ceil(count / nrow))
    canvas = np.full((ncol * height, nrow * width), 255, dtype=np.uint8)

    for i, image in enumerate(images):
        row = i // nrow
        col = i % nrow
        canvas[row * height : (row + 1) * height, col * width : (col + 1) * width] = image

    out = Image.fromarray(canvas)
    if clean:
        out = out.filter(ImageFilter.MedianFilter(size=3))
        arr = np.asarray(out.convert("L"), dtype=np.uint8)
        arr = np.where(arr < threshold, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr, mode="L")

    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clean 256x256 handwritten digits.")
    parser.add_argument("--checkpoint", default="checkpoints_256_clean/latest.pt")
    parser.add_argument(
        "--out",
        default="",
        help="Output path. If omitted, saves to cgan-images-256-clean/digit{digit}.png.",
    )
    parser.add_argument("--digit", type=int, default=7, choices=range(10))
    parser.add_argument("--num", type=int, default=16)
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--thickness", default="0.5")
    parser.add_argument("--slant", default="0.5")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--clean-threshold", type=int, default=210)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    generator = ConditionalGenerator(
        z_dim=int(checkpoint["z_dim"]),
        num_digits=int(checkpoint.get("num_digits", 10)),
        attr_dim=int(checkpoint.get("attr_dim", 2)),
        embed_dim=int(checkpoint.get("embed_dim", 64)),
        base_channels=int(checkpoint.get("base_channels", 48)),
    ).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    thickness = parse_float_list(args.thickness, args.num, 0.5)
    slant = parse_float_list(args.slant, args.num, 0.5)
    attrs = torch.tensor(
        [[np.clip(t, 0.0, 1.0), np.clip(s, 0.0, 1.0)] for t, s in zip(thickness, slant)],
        dtype=torch.float32,
        device=device,
    )
    digits = torch.full((args.num,), args.digit, dtype=torch.long, device=device)
    z = torch.randn(args.num, int(checkpoint["z_dim"]), device=device)

    with torch.no_grad():
        images = generator(z, digits, attrs)

    if args.out:
        out_path = Path(args.out)
    else:
        suffix = "_clean" if args.clean else ""
        out_path = Path("cgan-images-256-clean") / f"digit{args.digit}{suffix}.png"
    save_grid(images, out_path, args.nrow, clean=args.clean, threshold=args.clean_threshold)
    print(f"saved {args.num} digit={args.digit} to {out_path}")


if __name__ == "__main__":
    main()

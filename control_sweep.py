import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from cgan_model import ConditionalGenerator
from preprocess_digits import estimate_slant_deg, foreground_bbox


def parse_values(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def estimate_controls(image: np.ndarray, threshold: int = 245) -> tuple[float, float, float]:
    mask = image < threshold
    ink_pixels = int(mask.sum())
    ink_ratio = ink_pixels / float(image.size)
    bbox = foreground_bbox(image, threshold)
    if bbox is None:
        return 0.0, 0.0, 0.0
    top, left, bottom, right = bbox
    thickness = ink_pixels / float(max(1, bottom - top) + max(1, right - left))
    return ink_ratio, thickness, estimate_slant_deg(image, threshold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep thickness and slant controls for the clean 256 model.")
    parser.add_argument("--checkpoint", default="checkpoints_256_clean/latest.pt")
    parser.add_argument("--out", default="cgan-images-256-clean/control_sweep.png")
    parser.add_argument("--digit", type=int, default=3, choices=range(10))
    parser.add_argument("--thickness-values", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--slant-values", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--fixed-thickness", type=float, default=0.6)
    parser.add_argument("--fixed-slant", type=float, default=0.5)
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

    rows = [
        ("thickness", [[t, args.fixed_slant] for t in parse_values(args.thickness_values)]),
        ("slant", [[args.fixed_thickness, s] for s in parse_values(args.slant_values)]),
    ]
    z = torch.randn(1, int(checkpoint["z_dim"]), device=device)
    rendered_rows: list[list[np.ndarray]] = []
    stats_rows: list[list[tuple[float, float, float]]] = []

    for _, attr_values in rows:
        batch = len(attr_values)
        digits = torch.full((batch,), args.digit, dtype=torch.long, device=device)
        attrs = torch.tensor(attr_values, dtype=torch.float32, device=device).clamp(0.0, 1.0)
        with torch.no_grad():
            images = generator(z.repeat(batch, 1), digits, attrs)
        images = ((images.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
        row_images = images.squeeze(1).numpy()
        rendered_rows.append(list(row_images))
        stats_rows.append([estimate_controls(image) for image in row_images])

    cell = 256
    label_w = 90
    header_h = 34
    metric_h = 34
    row_h = cell + metric_h
    max_cols = max(len(values) for _, values in rows)
    canvas = Image.new("L", (label_w + max_cols * cell, header_h + len(rows) * row_h), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), f"digit {args.digit}", fill=0)

    for row_index, (name, attr_values) in enumerate(rows):
        y = header_h + row_index * row_h
        draw.text((4, y + 6), name, fill=0)
        for col, image in enumerate(rendered_rows[row_index]):
            x = label_w + col * cell
            canvas.paste(Image.fromarray(image), (x, y))
            value = attr_values[col][0 if name == "thickness" else 1]
            _, thickness, slant = stats_rows[row_index][col]
            draw.text((x + 4, y + cell + 2), f"in {value:.1f}", fill=0)
            draw.text((x + 4, y + cell + 16), f"t {thickness:.1f} s {slant:.0f}", fill=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"saved control sweep to {out_path}")


if __name__ == "__main__":
    main()

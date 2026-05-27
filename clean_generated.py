import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def clean_image(
    image: Image.Image,
    threshold: int,
    white_point: int,
    smooth: bool,
) -> Image.Image:
    gray = image.convert("L")
    if smooth:
        gray = gray.filter(ImageFilter.MedianFilter(size=3))

    arr = np.asarray(gray, dtype=np.uint8)
    cleaned = arr.copy()
    cleaned[cleaned >= white_point] = 255
    cleaned = np.where(cleaned < threshold, 0, 255).astype(np.uint8)
    return Image.fromarray(cleaned, mode="L")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean generated handwritten digit images for display.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=int, default=210)
    parser.add_argument("--white-point", type=int, default=245)
    parser.add_argument("--no-smooth", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.input)
    cleaned = clean_image(
        image,
        threshold=args.threshold,
        white_point=args.white_point,
        smooth=not args.no_smooth,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.save(output)
    print(f"saved cleaned image to {output}")


if __name__ == "__main__":
    main()

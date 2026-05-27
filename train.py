import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL import ImageChops
from PIL import ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cgan_model import ConditionalDiscriminator, ConditionalGenerator, IMAGE_SIZE, init_weights


class DigitDataset(Dataset):
    def __init__(
        self,
        images_path: str,
        writer_info_path: str,
        meta_path: str,
        binarize_real: bool = True,
        threshold: int = 210,
        thickness_augment: bool = True,
        thickness_steps: int = 4,
        slant_augment: bool = True,
        slant_strength: float = 0.45,
    ) -> None:
        self.images = np.load(images_path, mmap_mode="r")
        self.writer_info = np.load(writer_info_path)
        meta = np.load(meta_path)
        self.meta = meta["values"].astype(np.float32)
        self.binarize_real = binarize_real
        self.threshold = threshold
        self.thickness_augment = thickness_augment
        self.thickness_steps = thickness_steps
        self.slant_augment = slant_augment
        self.slant_strength = slant_strength

        if self.images.shape != (self.writer_info.shape[0], IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Expected images with shape (N,{IMAGE_SIZE},{IMAGE_SIZE}), got {self.images.shape}"
            )
        if self.images.shape[0] != self.meta.shape[0]:
            raise ValueError("Image count and preprocessing metadata count do not match.")

        self.digits = self.writer_info[:, 0].astype(np.int64)
        self.thickness = self.meta[:, 11].astype(np.float32)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.images[index]
        if self.binarize_real:
            image = np.where(image < self.threshold, 0, 255).astype(np.uint8)
        thickness = np.float32(np.random.rand() if self.thickness_augment else self.thickness[index])
        slant = np.float32(np.random.rand() if self.slant_augment else self.meta[index, 12])
        if self.slant_augment:
            image = shear_image(image, float(slant), self.slant_strength)
            if self.binarize_real:
                image = np.where(image < self.threshold, 0, 255).astype(np.uint8)
        if self.thickness_augment:
            image = adjust_thickness(image, float(thickness), self.thickness_steps)
            if self.binarize_real:
                image = np.where(image < self.threshold, 0, 255).astype(np.uint8)
        image = image.astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).unsqueeze(0)
        digit = torch.tensor(self.digits[index], dtype=torch.long)
        attrs = torch.tensor([thickness, slant], dtype=torch.float32)
        return image, digit, attrs


def shear_image(image: np.ndarray, slant_norm: float, strength: float) -> np.ndarray:
    shear = (slant_norm - 0.5) * 2.0 * strength
    pil_image = Image.fromarray(image)
    width, height = pil_image.size
    x_shift = abs(shear) * height
    new_width = int(round(width + x_shift))
    transformed = pil_image.transform(
        (new_width, height),
        Image.Transform.AFFINE,
        (1, shear, -x_shift if shear > 0 else 0, 0, 1, 0),
        resample=Image.Resampling.BILINEAR,
        fillcolor=255,
    )
    if new_width > width:
        left = (new_width - width) // 2
        transformed = transformed.crop((left, 0, left + width, height))
    elif new_width < width:
        canvas = Image.new("L", (width, height), 255)
        canvas.paste(transformed, ((width - new_width) // 2, 0))
        transformed = canvas
    bbox = ImageChops.invert(transformed).getbbox()
    if bbox is not None:
        center_x = (bbox[0] + bbox[2]) // 2
        shift = width // 2 - center_x
        transformed = ImageChops.offset(transformed, shift, 0)
        if shift > 0:
            transformed.paste(255, (0, 0, shift, height))
        elif shift < 0:
            transformed.paste(255, (width + shift, 0, width, height))
    return np.asarray(transformed, dtype=np.uint8)


def adjust_thickness(image: np.ndarray, thickness_norm: float, max_steps: int) -> np.ndarray:
    if max_steps <= 0:
        return image
    amount = (thickness_norm - 0.5) * 2.0
    steps = int(round(abs(amount) * max_steps))
    if steps == 0:
        return image

    pil_image = Image.fromarray(image)
    filter_cls = ImageFilter.MinFilter if amount > 0 else ImageFilter.MaxFilter
    for _ in range(steps):
        pil_image = pil_image.filter(filter_cls(size=3))
    return np.asarray(pil_image, dtype=np.uint8)


def save_grid(images: torch.Tensor, path: Path, nrow: int = 10) -> None:
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

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def sample_fixed_conditions(device: torch.device, z_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    digits = torch.arange(10, dtype=torch.long, device=device).repeat_interleave(6)
    thickness = torch.tensor([0.05, 0.5, 0.95, 0.5, 0.5, 0.5], device=device).repeat(10)
    slant = torch.tensor([0.5, 0.5, 0.5, 0.05, 0.5, 0.95], device=device).repeat(10)
    attrs = torch.stack([thickness, slant], dim=1)
    z = torch.randn(digits.size(0), z_dim, device=device)
    return z, digits, attrs


def binarization_loss(images: torch.Tensor) -> torch.Tensor:
    # Penalize mid-gray pixels. A clean black/white image has values near -1 or 1.
    return (1.0 - images.abs()).mean()


def ink_ratio(images: torch.Tensor) -> torch.Tensor:
    dark_prob = ((1.0 - images.clamp(-1, 1)) * 0.5).clamp(0, 1)
    return dark_prob.mean(dim=(1, 2, 3))


def background_loss(images: torch.Tensor, target_ink: torch.Tensor, margin: float) -> torch.Tensor:
    ratio = ink_ratio(images)
    return torch.relu(ratio - target_ink - margin).mean()


def mode_seeking_loss(
    generator: ConditionalGenerator,
    z: torch.Tensor,
    digits: torch.Tensor,
    attrs: torch.Tensor,
    images: torch.Tensor,
) -> torch.Tensor:
    z2 = torch.randn_like(z)
    images_2 = generator(z2, digits, attrs)
    image_distance = (images - images_2).abs().flatten(1).mean(dim=1)
    latent_distance = (z - z2).abs().mean(dim=1).clamp_min(1e-6)
    return 1.0 / (image_distance / latent_distance + 1e-5).mean()


def save_checkpoint(
    path: Path,
    generator: ConditionalGenerator,
    discriminator: ConditionalDiscriminator,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "g_optimizer": g_optimizer.state_dict(),
            "d_optimizer": d_optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "z_dim": args.z_dim,
            "num_digits": 10,
            "base_channels": args.base_channels,
            "embed_dim": args.embed_dim,
            "attr_dim": 2,
            "image_size": IMAGE_SIZE,
            "condition_keys": ["digit", "thickness_norm", "slant_norm"],
            "args": vars(args),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a clean 256x256 digit CGAN.")
    parser.add_argument("--images", default="data/Images_256.npy")
    parser.add_argument("--writer-info", default="data/WriterInfo_256.npy")
    parser.add_argument("--meta", default="data/PreprocessInfo_256.npz")
    parser.add_argument("--out-dir", default="checkpoints_256_clean")
    parser.add_argument("--sample-dir", default="cgan-images-256-clean")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--digit-loss-weight", type=float, default=4.0)
    parser.add_argument("--thickness-loss-weight", type=float, default=4.0)
    parser.add_argument("--slant-loss-weight", type=float, default=4.0)
    parser.add_argument("--binary-loss-weight", type=float, default=0.8)
    parser.add_argument("--background-loss-weight", type=float, default=1.5)
    parser.add_argument("--mode-seeking-weight", type=float, default=0.2)
    parser.add_argument("--ink-margin", type=float, default=0.03)
    parser.add_argument("--real-threshold", type=int, default=210)
    parser.add_argument("--no-binarize-real", action="store_true")
    parser.add_argument("--no-thickness-augment", action="store_true")
    parser.add_argument("--thickness-steps", type=int, default=4)
    parser.add_argument("--no-slant-augment", action="store_true")
    parser.add_argument("--slant-strength", type=float, default=0.45)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    dataset = DigitDataset(
        args.images,
        args.writer_info,
        args.meta,
        binarize_real=not args.no_binarize_real,
        threshold=args.real_threshold,
        thickness_augment=not args.no_thickness_augment,
        thickness_steps=args.thickness_steps,
        slant_augment=not args.no_slant_augment,
        slant_strength=args.slant_strength,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    generator = ConditionalGenerator(
        z_dim=args.z_dim,
        num_digits=10,
        attr_dim=2,
        embed_dim=args.embed_dim,
        base_channels=args.base_channels,
    ).to(device)
    discriminator = ConditionalDiscriminator(
        num_digits=10,
        attr_dim=2,
        embed_dim=512,
        base_channels=args.base_channels,
    ).to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    start_epoch = 1
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["step"])

    out_dir = Path(args.out_dir)
    sample_dir = Path(args.sample_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(
        json.dumps({**vars(args), "image_size": IMAGE_SIZE, "num_samples": len(dataset)}, indent=2),
        encoding="utf-8",
    )

    fixed_z, fixed_digits, fixed_attrs = sample_fixed_conditions(device, args.z_dim)

    adversarial_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    for epoch in range(start_epoch, args.epochs + 1):
        for real_images, digits, attrs in dataloader:
            real_images = real_images.to(device, non_blocking=True)
            digits = digits.to(device, non_blocking=True)
            attrs = attrs.to(device, non_blocking=True)
            batch_size = real_images.size(0)
            real_ink = ink_ratio(real_images).detach()

            real_targets = torch.empty(batch_size, 1, device=device).uniform_(0.85, 1.0)
            fake_targets = torch.empty(batch_size, 1, device=device).uniform_(0.0, 0.15)

            real_score, real_digit_logits, real_attr_pred = discriminator(real_images, return_aux=True)
            real_attr_weights = torch.tensor(
                [args.thickness_loss_weight, args.slant_loss_weight],
                dtype=torch.float32,
                device=device,
            )
            d_attr_loss = ((real_attr_pred - attrs) ** 2 * real_attr_weights).mean()
            d_aux_loss = args.digit_loss_weight * ce_loss(real_digit_logits, digits) + d_attr_loss

            z = torch.randn(batch_size, args.z_dim, device=device)
            with torch.no_grad():
                fake_images = generator(z, digits, attrs)
            fake_score = discriminator(fake_images.detach())
            d_loss = adversarial_loss(real_score, real_targets) + adversarial_loss(fake_score, fake_targets) + d_aux_loss

            d_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            d_optimizer.step()

            z = torch.randn(batch_size, args.z_dim, device=device)
            fake_images = generator(z, digits, attrs)
            fake_score, fake_digit_logits, fake_attr_pred = discriminator(fake_images, return_aux=True)
            g_attr_loss = ((fake_attr_pred - attrs) ** 2 * real_attr_weights).mean()
            g_aux_loss = args.digit_loss_weight * ce_loss(fake_digit_logits, digits) + g_attr_loss
            binary = binarization_loss(fake_images)
            background = background_loss(fake_images, real_ink, args.ink_margin)
            if args.mode_seeking_weight > 0:
                ms = mode_seeking_loss(generator, z, digits, attrs, fake_images)
            else:
                ms = torch.tensor(0.0, device=device)

            g_loss = (
                adversarial_loss(fake_score, torch.ones_like(fake_score))
                + g_aux_loss
                + args.binary_loss_weight * binary
                + args.background_loss_weight * background
                + args.mode_seeking_weight * ms
            )

            g_optimizer.zero_grad(set_to_none=True)
            g_loss.backward()
            g_optimizer.step()

            if global_step % 50 == 0:
                with torch.no_grad():
                    digit_acc = (fake_digit_logits.argmax(dim=1) == digits).float().mean()
                    fake_ink = ink_ratio(fake_images).mean()
                    real_ink_mean = real_ink.mean()
                print(
                    f"epoch={epoch:03d} step={global_step:06d} "
                    f"d_loss={d_loss.item():.4f} g_loss={g_loss.item():.4f} "
                    f"d_aux={d_aux_loss.item():.4f} g_aux={g_aux_loss.item():.4f} "
                    f"bin={binary.item():.4f} bg={background.item():.4f} "
                    f"ms={ms.item():.4f} acc={digit_acc.item():.3f} "
                    f"ink={fake_ink.item():.3f}/{real_ink_mean.item():.3f}"
                )

            if global_step % args.sample_every == 0:
                generator.eval()
                with torch.no_grad():
                    samples = generator(fixed_z, fixed_digits, fixed_attrs)
                save_grid(samples, sample_dir / f"sample_{global_step:06d}.png", nrow=10)
                generator.train()

            global_step += 1
            if args.max_steps > 0 and global_step >= args.max_steps:
                save_checkpoint(out_dir / "latest.pt", generator, discriminator, g_optimizer, d_optimizer, epoch, global_step, args)
                return

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(out_dir / f"cgan_epoch_{epoch:03d}.pt", generator, discriminator, g_optimizer, d_optimizer, epoch, global_step, args)
            save_checkpoint(out_dir / "latest.pt", generator, discriminator, g_optimizer, d_optimizer, epoch, global_step, args)


if __name__ == "__main__":
    main()

import torch
from torch import nn
from torch.nn.utils import spectral_norm


IMAGE_SIZE = 256


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        z_dim: int,
        num_digits: int = 10,
        attr_dim: int = 2,
        embed_dim: int = 64,
        base_channels: int = 48,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.num_digits = num_digits
        self.attr_dim = attr_dim
        self.embed_dim = embed_dim
        self.base_channels = base_channels

        self.digit_embedding = nn.Embedding(num_digits, embed_dim)
        self.attr_encoder = nn.Sequential(
            nn.Linear(attr_dim, embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.project = nn.Sequential(
            nn.Linear(z_dim + embed_dim * 2, base_channels * 8 * 4 * 4),
            nn.BatchNorm1d(base_channels * 8 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.net = nn.Sequential(
            *self._resize_conv_block(base_channels * 8, base_channels * 8),
            *self._resize_conv_block(base_channels * 8, base_channels * 4),
            *self._resize_conv_block(base_channels * 4, base_channels * 2),
            *self._resize_conv_block(base_channels * 2, base_channels),
            *self._resize_conv_block(base_channels, max(base_channels // 2, 8)),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(max(base_channels // 2, 8), 1, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    @staticmethod
    def _resize_conv_block(in_channels: int, out_channels: int) -> list[nn.Module]:
        return [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

    def forward(
        self,
        z: torch.Tensor,
        digits: torch.Tensor,
        attrs: torch.Tensor,
    ) -> torch.Tensor:
        cond = torch.cat(
            [
                self.digit_embedding(digits),
                self.attr_encoder(attrs),
            ],
            dim=1,
        )
        x = torch.cat([z, cond], dim=1)
        x = self.project(x).view(x.size(0), self.base_channels * 8, 4, 4)
        return self.net(x)


class ConditionalDiscriminator(nn.Module):
    def __init__(
        self,
        num_digits: int = 10,
        attr_dim: int = 2,
        embed_dim: int = 512,
        base_channels: int = 48,
    ) -> None:
        super().__init__()
        self.num_digits = num_digits
        self.attr_dim = attr_dim
        self.embed_dim = embed_dim
        self.base_channels = base_channels

        self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(1, base_channels, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 8, base_channels * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 8, base_channels * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        feature_dim = base_channels * 8 * 4 * 4
        self.to_embedding = spectral_norm(nn.Linear(feature_dim, embed_dim))
        self.score = spectral_norm(nn.Linear(embed_dim, 1))
        self.digit_head = spectral_norm(nn.Linear(embed_dim, num_digits))
        self.attr_head = spectral_norm(nn.Linear(embed_dim, attr_dim))

    def forward(
        self,
        images: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.features(images).flatten(1)
        x = self.to_embedding(x)
        score = self.score(x)
        if not return_aux:
            return score
        return score, self.digit_head(x), self.attr_head(x)

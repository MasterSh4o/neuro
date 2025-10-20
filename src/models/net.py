from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm_type: str, num_channels: int, gn_groups: int) -> nn.Module:
    norm_type_l = norm_type.lower()
    if norm_type_l in {"batchnorm", "bn", "batchnorm2d"}:
        return nn.BatchNorm2d(num_channels)
    if norm_type_l in {"groupnorm", "gn"}:
        groups = max(1, min(gn_groups, num_channels))
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"Unsupported norm type: {norm_type}")


@dataclass
class BlockConfig:
    in_channels: int
    out_channels: int
    stride: int
    repeats: int


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x, residual):
        if not self.training or self.drop_prob <= 0.0:
            return x + residual
        keep_prob = 1.0 - self.drop_prob
        noise = torch.empty(
            x.shape[0],
            *((1,) * (x.ndim - 1)),
            device=x.device,
            dtype=x.dtype,
        ).bernoulli_(keep_prob)
        return x + residual * noise / keep_prob


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        inner = max(channels // reduction, 1)
        self.fc1 = nn.Conv2d(channels, inner, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(inner, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm_type: str,
        gn_groups: int,
        drop_path: float,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = _make_norm(norm_type, out_channels, gn_groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = _make_norm(norm_type, out_channels, gn_groups)
        self.se = SEBlock(out_channels)
        self.activation = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _make_norm(norm_type, out_channels, gn_groups),
            )
        else:
            self.shortcut = nn.Identity()
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.se(x)
        shortcut = self.shortcut(residual)
        if self.drop_path is not None:
            x = self.drop_path(x, shortcut)
        else:
            x = x + shortcut
        return self.activation(x)


class InterferoNetMultiLabel(nn.Module):
    def __init__(
        self,
        out_dim: int,
        base_channels: int = 32,
        width_multipliers: Sequence[int] = (2, 4, 8, 12),
        block_repeats: Sequence[int] = (1, 1, 1, 1),
        norm_type: str = "groupnorm",
        gn_groups: int = 16,
        stochastic_depth: float = 0.0,
        dropout: float = 0.2,
    ):
        super().__init__()
        width_multipliers = tuple(width_multipliers)
        block_repeats = tuple(block_repeats)
        if len(width_multipliers) != len(block_repeats):
            raise ValueError("width_multipliers and block_repeats must have the same length.")
        self.out_dim = out_dim
        self.base_channels = base_channels

        stem_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
            _make_norm(norm_type, stem_channels, gn_groups),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        blocks: List[nn.Module] = []
        in_c = stem_channels
        total_blocks = sum(block_repeats)
        block_idx = 0
        for multiplier, repeats in zip(width_multipliers, block_repeats):
            out_c = base_channels * multiplier
            for i in range(repeats):
                stride = 2 if (i == 0 and (in_c != out_c or block_idx > 0)) else 1
                drop_path_rate = stochastic_depth * block_idx / max(1, total_blocks - 1)
                blocks.append(
                    ResidualBlock(
                        in_channels=in_c,
                        out_channels=out_c,
                        stride=stride,
                        norm_type=norm_type,
                        gn_groups=gn_groups,
                        drop_path=drop_path_rate,
                    )
                )
                in_c = out_c
                block_idx += 1
        self.backbone = nn.Sequential(*blocks)
        self.head_norm = _make_norm(norm_type, in_c, gn_groups)
        self.head_act = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_c, in_c),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_c, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool(x)
        x = self.backbone(x)
        x = self.head_act(self.head_norm(x))
        x = self.global_pool(x).flatten(1)
        x = self.classifier(x)
        return x

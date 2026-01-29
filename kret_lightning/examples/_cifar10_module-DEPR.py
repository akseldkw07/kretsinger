import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F


from kret_lightning.mixin_metrics import MetricMixin
from kret_lightning.mixin_callbacks import CallbackMixin
from kret_lightning.base_lightning_nn import BaseLightningNN
from kret_lightning.abc_lightning import HPasKwargs

# Presets for CIFAR-10 (3 layer groups, suitable for 32x32 images)
# Each list defines [blocks_in_layer1, blocks_in_layer2, blocks_in_layer3]
# Total conv layers in residual portion = sum(blocks) * 2
RESNET_PRESETS: dict[str, list[int]] = {
    "tiny": [1, 1, 1],  # 6 conv layers, ~0.3M params
    "small": [2, 2, 2],  # 12 conv layers, ~0.6M params
    "medium": [3, 3, 3],  # 18 conv layers, ~0.9M params
    "large": [3, 4, 6],  # 26 conv layers, ~1.3M params
    "xlarge": [3, 6, 9],  # 36 conv layers, ~1.8M params
}

PresetLiteral = t.Literal["tiny", "small", "medium", "large", "xlarge"]


class ResidualBlock(nn.Module):
    """
    Simple residual block
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet-like architecture for CIFAR-10 with preset configurations."""

    def __init__(self, preset: PresetLiteral = "small", num_filters: int = 64, dropout_rate: float = 0.3):
        super().__init__()
        if preset not in RESNET_PRESETS:
            raise ValueError(f"Unknown preset {preset!r}. Choose from: {list(RESNET_PRESETS.keys())}")

        blocks_per_layer = RESNET_PRESETS[preset]
        self.preset = preset
        self.blocks_per_layer = blocks_per_layer
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate

        # Initial conv layer
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.dropout = nn.Dropout(dropout_rate)

        # Build residual layer groups based on preset
        # First layer: no downsampling (stride=1), subsequent layers: downsample (stride=2)
        # Channels double at each layer after the first
        self.res_layers = nn.ModuleList()
        in_ch = num_filters
        for i, num_blocks in enumerate(blocks_per_layer):
            out_ch = num_filters * (2**i)
            stride = 1 if i == 0 else 2
            self.res_layers.append(self._make_layer(in_ch, out_ch, num_blocks, stride))
            in_ch = out_ch

        # Classification head (in_ch is the final output channels)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, 10)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        for layer in self.res_layers:
            out = layer(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CIFAR10ResNet(MetricMixin, BaseLightningNN, CallbackMixin):
    """Lightning wrapper for ResNet on CIFAR-10"""

    _criterion = nn.CrossEntropyLoss()

    def __init__(
        self,
        preset: PresetLiteral = "small",
        num_filters: int = 64,
        dropout_rate: float = 0.3,
        num_classes: int = 10,
        **kwargs: t.Unpack[HPasKwargs],
    ):
        super().__init__(**kwargs)

        print(f"Saving hparams, ignoring {self.ignore_hparams}")
        self.save_hyperparameters(ignore=self.ignore_hparams)
        self.setup_metrics(task="multiclass", num_classes=num_classes)

        self.model = ResNet(preset=preset, num_filters=num_filters, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

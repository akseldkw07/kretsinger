import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy

from kret_lightning import *


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
    """ResNet-like architecture for CIFAR-10"""

    def __init__(self, num_blocks: int = 3, num_filters: int = 64, dropout_rate: float = 0.5):
        super().__init__()
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate

        # Initial conv layer
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.dropout = nn.Dropout(dropout_rate)

        # Residual blocks
        self.layer1 = self._make_layer(num_filters, num_filters, num_blocks, stride=1)
        self.layer2 = self._make_layer(num_filters, num_filters * 2, num_blocks, stride=2)
        self.layer3 = self._make_layer(num_filters * 2, num_filters * 4, num_blocks, stride=2)

        # Classification head
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_filters * 4, 10)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CIFAR10ResNet(BaseLightningNN, CallbackMixin, MetricMixin):
    """Lightning wrapper for ResNet on CIFAR-10"""

    _criterion = nn.CrossEntropyLoss()

    def __init__(
        self,
        num_blocks: int = 3,
        num_filters: int = 64,
        dropout_rate: float = 0.5,
        **kwargs: t.Unpack[HPasKwargs],
    ):
        super().__init__(**kwargs)

        self.model = ResNet(num_blocks=num_blocks, num_filters=num_filters, dropout_rate=dropout_rate)

        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        self.train_accuracy(logits, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        self.log("train_acc", self.train_accuracy, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        self.val_accuracy(logits, y)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        self.test_accuracy(logits, y)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True)

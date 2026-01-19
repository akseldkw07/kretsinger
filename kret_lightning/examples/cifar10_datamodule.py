from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: Path | str, batch_size: int = 128, num_workers: int = 4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        # CIFAR-10 normalization constants
        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        self.test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )

        self.data_loader_args = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "persistent_workers": True,
        }

    def prepare_data(self):
        # Download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.train_transform)
            self.cifar10_train, self.cifar10_val = random_split(
                cifar10_full, [45000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.test_transform)

        if stage == "predict":
            self.cifar10_predict = CIFAR10(self.data_dir, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, **self.data_loader_args, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, **self.data_loader_args)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, **self.data_loader_args)

    def predict_dataloader(self):
        return DataLoader(self.cifar10_predict, **self.data_loader_args)

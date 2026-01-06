import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size: int = 64, shuffle: bool = True, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        self.data_loader_args = {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "persistent_workers": True,
        }

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, **self.data_loader_args)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, **self.data_loader_args)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, **self.data_loader_args)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, **self.data_loader_args)

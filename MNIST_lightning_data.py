import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT, EVAL_DATALOADERS, TRAIN_DATALOADERS

class MNISTDataModule(pl.LightningDataModule):

    def prepare_data(self) -> None:
        # prepare transforms standard to MNIST
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
            
        mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transform)
        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])

        mnist_train = DataLoader(mnist_train, batch_size=64)
        return mnist_train

    def val_dataloader(self) -> EVAL_DATALOADERS:
        mnist_val = DataLoader(self.mnist_val, batch_size=64)
        return mnist_val
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        mnist_test = MNIST(os.getcwd(), train=False, download=False, transforms=transform)
        mnist_test = DataLoader(mnist_test, batch_size=64)
        return mnist_test
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip

from mydataset import MyDataSet


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = "/home/bennie/bennie/voc/VOC2007"
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # shi.MNIST(self.data_dir, train=True, download=True)
        # datasets.MNIST(self.data_dir, train=False, download=True)

        # ??????????????????????????

        MyDataSet(self.data_dir, train=True)
        MyDataSet(self.data_dir, train=False)



    def setup(self, stage):
        # entire_dataset = datasets.MNIST(
        #     root=self.data_dir,
        #     train=True,
        #     transform=transforms.Compose([
        #         transforms.RandomVerticalFlip(),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #     ]),
        #     download=False,
        # )
        # self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000])
        # self.test_ds = datasets.MNIST(
        #     root=self.data_dir,
        #     train=False,
        #     transform=transforms.ToTensor(),
        #     download=False,
        # )
        self.train_ds = MyDataSet(self.data_dir, train=True)
        self.val_ds = MyDataSet(self.data_dir, train=False)
        self.test_ds = self.val_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

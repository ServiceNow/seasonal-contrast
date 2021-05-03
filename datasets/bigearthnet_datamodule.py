import os

from PIL import Image
from torchvision import transforms
from pytorch_lightning import LightningDataModule

from datasets.bigearthnet_dataset import Bigearthnet
from utils.data import random_subset, LMDBDataset, InfiniteDataLoader


class BigearthnetDataModule(LightningDataModule):

    def __init__(self, data_dir, bands=None, train_frac=None, val_frac=None, lmdb=False, batch_size=32, num_workers=16, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.bands = bands
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.lmdb = lmdb
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None

    @property
    def num_classes(self):
        return 19

    def setup(self, stage=None):
        train_transforms = self.train_transform() if self.train_transforms is None else self.train_transforms
        if self.lmdb:
            self.train_dataset = LMDBDataset(
                lmdb_file=os.path.join(self.data_dir, 'train.lmdb'),
                transform=train_transforms
            )
        else:
            self.train_dataset = Bigearthnet(
                root=self.data_dir,
                split='train',
                bands=self.bands,
                transform=train_transforms
            )
        if self.train_frac is not None and self.train_frac < 1:
            self.train_dataset = random_subset(self.train_dataset, self.train_frac, self.seed)

        val_transforms = self.val_transform() if self.val_transforms is None else self.val_transforms
        if self.lmdb:
            self.val_dataset = LMDBDataset(
                lmdb_file=os.path.join(self.data_dir, 'val.lmdb'),
                transform=val_transforms
            )
        else:
            self.val_dataset = Bigearthnet(
                root=self.data_dir,
                split='val',
                bands=self.bands,
                transform=val_transforms
            )
        if self.val_frac is not None and self.val_frac < 1:
            self.val_dataset = random_subset(self.val_dataset, self.val_frac, self.seed)

    @staticmethod
    def train_transform():
        return transforms.Compose([
            transforms.Resize((128, 128), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

    @staticmethod
    def val_transform():
        return transforms.Compose([
            transforms.Resize((128, 128), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

    def train_dataloader(self):
        return InfiniteDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return InfiniteDataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

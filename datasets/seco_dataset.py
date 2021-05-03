import os
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pl_bolts.models.self_supervised.moco.transforms import GaussianBlur, imagenet_normalization

ALL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
RGB_BANDS = ['B4', 'B3', 'B2']

QUANTILES = {
    'min_q': {
        'B2': 3.0,
        'B3': 2.0,
        'B4': 0.0
    },
    'max_q': {
        'B2': 88.0,
        'B3': 103.0,
        'B4': 129.0
    }
}


class SeasonalContrastBase(Dataset):

    def __init__(self, root, bands=None, transform=None):
        super().__init__()
        self.root = Path(root)
        self.bands = bands if bands is not None else RGB_BANDS
        self.transform = transform

        self._samples = None

    @property
    def samples(self):
        if self._samples is None:
            self._samples = self.get_samples()
        return self._samples

    def get_samples(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


def normalize(img, min_q, max_q):
    img = (img - min_q) / (max_q - min_q)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def read_image(path, bands, quantiles=None):
    channels = []
    for b in bands:
        ch = rasterio.open(path / f'{b}.tif').read(1)
        if quantiles is not None:
            ch = normalize(ch, min_q=quantiles['min_q'][b], max_q=quantiles['max_q'][b])
        channels.append(ch)
    img = np.dstack(channels)
    img = Image.fromarray(img)
    return img


class SeasonalContrastBasic(SeasonalContrastBase):

    def get_samples(self):
        # return [path for path in self.root.glob('*/*') if path.is_dir()]
        samples = []
        for entry in os.scandir(self.root):
            for subentry in os.scandir(entry.path):
                if subentry.is_dir():
                    samples.append(Path(subentry.path))
        return samples

    def __getitem__(self, index):
        path = self.samples[index]
        img = read_image(path, self.bands, QUANTILES)
        if self.transform is not None:
            img = self.transform(img)
        return img


class SeasonalContrastTemporal(SeasonalContrastBase):

    def get_samples(self):
        return [path for path in self.root.glob('*') if path.is_dir()]

    def __getitem__(self, index):
        root = self.samples[index]
        paths = np.random.choice([path for path in root.glob('*') if path.is_dir()], 2)
        images = []
        for path in paths:
            img = read_image(path, self.bands, QUANTILES)
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
        return images[0], images[1]


class SeasonalContrastMultiAug(SeasonalContrastBase):

    augment = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
    ])
    preprocess = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.ToTensor(),
        imagenet_normalization()
    ])

    def get_samples(self):
        return [path for path in self.root.glob('*') if path.is_dir()]

    def __getitem__(self, index):
        root = self.samples[index]
        sorted_paths = sorted([path for path in root.glob('*') if path.is_dir()], reverse=True)
        t0, t1, t2 = [read_image(path, self.bands, QUANTILES) for path in np.random.choice(sorted_paths, 3)]

        q = t0
        k0 = self.augment(t1)
        k1 = t2
        k2 = self.augment(t0)

        q = self.preprocess(q)
        k0 = self.preprocess(k0)
        k1 = self.preprocess(k1)
        k2 = self.preprocess(k2)

        return q, [k0, k1, k2]

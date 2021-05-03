import argparse

import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange

from datasets.seco_dataset import SeasonalContrastBasic, RGB_BANDS


def compute_quantiles(root, bands, num_samples):
    dataset = SeasonalContrastBasic(root, bands)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, collate_fn=lambda x: x[0])

    imgs = []
    for _ in trange(num_samples):
        imgs.append(np.array(next(iter(loader))).reshape(-1, len(bands)))
    imgs = np.concatenate(imgs, axis=0)

    quantiles = {'min_q': {}, 'max_q': {}}
    for i, b in enumerate(bands):
        values = imgs[:, i].reshape(-1)
        quantiles['min_q'][b] = np.quantile(values, 0.02)
        quantiles['max_q'][b] = np.quantile(values, 0.98)

    return quantiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--num_samples', type=int, default=3000)
    args = parser.parse_args()

    quantiles = compute_quantiles(
        root=args.data_dir,
        bands=RGB_BANDS,
        num_samples=args.num_samples
    )
    print(quantiles)

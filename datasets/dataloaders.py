import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

from config.config import Config
from datasets.casia import Casia2DetLocDataset, Casia1DetDataset

def set_seed(seed: int):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_dataloaders(config: Config):
    # train on CASIA2
    ds2 = Casia2DetLocDataset(config.casia2_root, img_size=config.img_size)
    ds1 = Casia1DetDataset(config.casia1_root, img_size=config.img_size)
    n_ds2 = len(ds2)

    idx_all = np.arange(n_ds2)
    np.random.shuffle(idx_all)
    n_train = int(0.7 * n_ds2)
    n_val = int(0.15 * n_ds2)
    n_test = n_ds2 - n_train - n_val
    idx_train = idx_all[:n_train]
    idx_val = idx_all[n_train:n_train + n_val]
    idx_test = idx_all[n_train + n_val:]

    ds2_train = Subset(ds2, idx_train.tolist())
    ds2_val = Subset(ds2, idx_val.tolist())
    ds2_test = Subset(ds2, idx_test.tolist())

    #train
    train_dataset = ds2_train
    val_dataset = ds2_val
    test_c2_dataset = ds2_test
    test_c1_dataset = ds1

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    test_c1_loader = DataLoader(
        test_c1_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    test_c2_loader = DataLoader(
        test_c2_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, val_loader, test_c1_loader, test_c2_loader

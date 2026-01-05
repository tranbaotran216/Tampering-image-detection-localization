import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

from config.config import Config
from datasets.casia import Casia2DetLocDataset, Casia1DetDataset
from datasets.coverage import CoverageDetLocDataset, coverage_base_id_from_path
from datasets.columbia import ColumbiaDetLocDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def build_dataloaders(config: Config):
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

    train_loader = _make_loader(ds2_train, config.batch_size, True, config.num_workers, True)
    val_loader = _make_loader(ds2_val, config.batch_size, False, config.num_workers, False)
    test_c1_loader = _make_loader(ds1, config.batch_size, False, config.num_workers, False)
    test_c2_loader = _make_loader(ds2_test, config.batch_size, False, config.num_workers, False)

    return train_loader, val_loader, test_c1_loader, test_c2_loader


def split_coverage_by_base_id(ds: CoverageDetLocDataset, val_ratio: float, seed: int):
    rng = np.random.RandomState(int(seed))
    id_to_idxs = {}
    ids = []

    for i, (img_path, _, _, _) in enumerate(ds.samples):
        base = coverage_base_id_from_path(img_path)
        if base not in id_to_idxs:
            id_to_idxs[base] = []
            ids.append(base)
        id_to_idxs[base].append(i)

    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * float(val_ratio))))
    val_ids = set(ids[:n_val])

    tr_idx = []
    va_idx = []
    for base, idxs in id_to_idxs.items():
        if base in val_ids:
            va_idx.extend(idxs)
        else:
            tr_idx.extend(idxs)

    return Subset(ds, tr_idx), Subset(ds, va_idx)


def build_coverage_dataloaders(config: Config, val_ratio: float = 0.2, group_split: bool = True):
    if not hasattr(config, "coverage_root"):
        raise AttributeError("Config missing coverage_root")

    ds = CoverageDetLocDataset(config.coverage_root, img_size=config.img_size)

    if group_split:
        ds_tr, ds_va = split_coverage_by_base_id(ds, val_ratio=val_ratio, seed=config.seed)
    else:
        n = len(ds)
        n_val = max(1, int(round(n * float(val_ratio))))
        n_tr = max(1, n - n_val)
        g = torch.Generator()
        g.manual_seed(int(config.seed))
        ds_tr, ds_va = torch.utils.data.random_split(ds, [n_tr, n_val], generator=g)

    tr_loader = _make_loader(ds_tr, config.batch_size, True, config.num_workers, True)
    va_loader = _make_loader(ds_va, config.batch_size, False, config.num_workers, False)
    return tr_loader, va_loader


def build_columbia_dataloaders(config: Config, val_ratio: float = 0.2, dilate_k: int = 9):
    if not hasattr(config, "columbia_root"):
        raise AttributeError("Config missing columbia_root")

    ds = ColumbiaDetLocDataset(config.columbia_root, img_size=config.img_size, dilate_k=dilate_k)

    n = len(ds)
    n_val = max(1, int(round(n * float(val_ratio))))
    n_tr = max(1, n - n_val)
    g = torch.Generator()
    g.manual_seed(int(config.seed))
    ds_tr, ds_va = torch.utils.data.random_split(ds, [n_tr, n_val], generator=g)

    tr_loader = _make_loader(ds_tr, config.batch_size, True, config.num_workers, True)
    va_loader = _make_loader(ds_va, config.batch_size, False, config.num_workers, False)
    return tr_loader, va_loader


def build_cov_col_mix_dataloaders(config: Config, val_ratio: float = 0.2, dilate_k: int = 9, group_split_coverage: bool = True):
    cov_tr, cov_va = build_coverage_dataloaders(config, val_ratio=val_ratio, group_split=group_split_coverage)
    col_tr, col_va = build_columbia_dataloaders(config, val_ratio=val_ratio, dilate_k=dilate_k)

    mix_tr = ConcatDataset([cov_tr.dataset, col_tr.dataset])
    mix_va = ConcatDataset([cov_va.dataset, col_va.dataset])

    mix_tr_loader = _make_loader(mix_tr, config.batch_size, True, config.num_workers, True)
    mix_va_loader = _make_loader(mix_va, config.batch_size, False, config.num_workers, False)

    return {
        "coverage_train": cov_tr,
        "coverage_val": cov_va,
        "columbia_train": col_tr,
        "columbia_val": col_va,
        "mix_train": mix_tr_loader,
        "mix_val": mix_va_loader,
    }

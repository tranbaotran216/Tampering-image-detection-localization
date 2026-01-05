import os
import argparse
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

from datasets.coverage import CoverageDetLocDataset
from datasets.columbia import ColumbiaDetLocDataset


def parse_args():
    p = argparse.ArgumentParser("Debug Coverage/Columbia loaders")
    p.add_argument("--coverage-root", type=str, default="./data/COVERAGE")
    p.add_argument("--columbia-root", type=str, default="./data/COLUMBIA")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--save-dir", type=str, default="./debug_out")
    return p.parse_args()


def _to_u8_mask(m: torch.Tensor):
    m = m.detach().cpu().float()
    if m.ndim == 3:
        m = m[0]
    m = (m > 0.5).numpy().astype(np.uint8) * 255
    return m


def _to_u8_img(x: torch.Tensor):
    x = x.detach().cpu().float()
    x = x.clamp(0, 1)
    x = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return x


def summarize_dataset(ds, name: str):
    n = len(ds)
    labels = []
    has_masks = []
    for i in range(min(n, 2000)):
        s = ds.samples[i]
        labels.append(int(s[2]))
        has_masks.append(bool(s[3]))
    labels = np.array(labels, dtype=np.int64)
    has_masks = np.array(has_masks, dtype=np.bool_)
    print(f"\n[{name}] n={n}")
    if labels.size > 0:
        print(f"  label0={int((labels==0).sum())} label1={int((labels==1).sum())} (sampled {labels.size})")
        print(f"  has_mask_true={int(has_masks.sum())} has_mask_false={int((~has_masks).sum())}")


def inspect_samples(ds, name: str, save_dir: str, n_samples: int):
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nInspect {name} (first {n_samples} samples):")
    for i in range(min(len(ds), n_samples)):
        item = ds[i]
        img = item["image"]
        mask = item["mask"]
        label = item["label"]
        has_mask = item["has_mask"]
        path = item.get("path", "")

        print(
            f"  idx={i:04d} label={int(label.item())} has_mask={bool(has_mask.item())} "
            f"img={tuple(img.shape)} mask={tuple(mask.shape)} "
            f"mask_mean={float(mask.float().mean().item()):.4f} path={os.path.basename(path)}"
        )

        img_u8 = _to_u8_img(img)
        mask_u8 = _to_u8_mask(mask)

        Image.fromarray(img_u8).save(os.path.join(save_dir, f"{name}_{i:04d}_img.png"))
        Image.fromarray(mask_u8).save(os.path.join(save_dir, f"{name}_{i:04d}_mask.png"))


def inspect_batch(ds, name: str, batch_size: int, num_workers: int):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=False)
    batch = next(iter(dl))
    print(f"\nBatch check [{name}]")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} min={float(v.min()):.3f} max={float(v.max()):.3f}")
        else:
            print(f"  {k}: type={type(v)}")


def main():
    args = parse_args()

    cov = CoverageDetLocDataset(args.coverage_root, img_size=args.img_size)
    col = ColumbiaDetLocDataset(args.columbia_root, img_size=args.img_size, dilate_k=9, include_auth=False)

    summarize_dataset(cov, "COVERAGE")
    summarize_dataset(col, "COLUMBIA")

    inspect_samples(cov, "coverage", args.save_dir, args.n_samples)
    inspect_samples(col, "columbia", args.save_dir, args.n_samples)

    inspect_batch(cov, "COVERAGE", args.batch_size, args.num_workers)
    inspect_batch(col, "COLUMBIA", args.batch_size, args.num_workers)

    print(f"\nSaved debug images to: {args.save_dir}")


if __name__ == "__main__":
    main()

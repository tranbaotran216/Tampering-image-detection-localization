import os
import csv
import argparse
from time import time
import torch.nn as nn
import numpy as np
from PIL import Image
from datasets.coverage import CoverageDetLocDataset
from datasets.columbia import ColumbiaDetLocDataset
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler, random_split
import torchvision.transforms as T
from tqdm import tqdm

from config.config import Config
from models.swint2 import SwinT2_SRM_FPN_DetLoc
from losses import multi_task_loss
from eval_utils import evaluate
from datasets.dataloaders import build_dataloaders, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Finetune SwinT2 on COVERAGE and Columbia (Schedule B)")
    p.add_argument("--config", type=str, default="./config/config_swint2.yaml")
    p.add_argument("--init-ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--coverage-root", type=str, default="./data/COVERAGE")
    p.add_argument("--columbia-root", type=str, default="./data/COLUMBIA")
    p.add_argument("--epochs-coverage", type=int, default=10)
    p.add_argument("--epochs-columbia", type=int, default=10)
    p.add_argument("--epochs-mix", type=int, default=5)
    p.add_argument("--casia-ratio", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--freeze-backbone-epochs", type=int, default=2)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def _get(cfg, k, default=None):
    if isinstance(cfg, dict):
        return cfg.get(k, default)
    if hasattr(cfg, k):
        return getattr(cfg, k)
    return default


def build_img_tf(img_size: int):
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])


def build_mask_tf(img_size: int):
    return T.Compose([T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST), T.PILToTensor()])


def _mask_to_01(mask_u8: torch.Tensor):
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[:1]
    m = mask_u8.float()
    if m.max() > 1.0:
        m = m / 255.0
    m = (m > 0.5).float()
    return m

def collate_detloc(batch):
    images = torch.stack([b["image"] for b in batch], 0)
    masks = torch.stack([b["mask"] for b in batch], 0)

    labels = []
    has_masks = []
    paths = []

    for b in batch:
        y = b["label"]
        if torch.is_tensor(y):
            y = y.view(-1)
            y = y[0].float() if y.numel() > 0 else torch.tensor(float(y.item()), dtype=torch.float32)
        else:
            y = torch.tensor(float(y), dtype=torch.float32)
        labels.append(y)

        hm = b["has_mask"]
        if torch.is_tensor(hm):
            hm = bool(hm.view(-1)[0].item()) if hm.numel() > 0 else bool(hm.item())
        else:
            hm = bool(hm)
        has_masks.append(torch.tensor(hm, dtype=torch.bool))

        if "path" in b:
            paths.append(b["path"])

    labels = torch.stack(labels, 0)
    has_masks = torch.stack(has_masks, 0)

    out = {"image": images, "mask": masks, "label": labels, "has_mask": has_masks}
    if len(paths) == len(batch):
        out["path"] = paths
    return out

def make_mixed_loader(datasets, probs, batch_size, num_workers, seed):
    concat = ConcatDataset(datasets)
    weights = []
    for ds, p in zip(datasets, probs):
        n = len(ds)
        w = (p / max(n, 1)) * np.ones(n, dtype=np.float32)
        weights.append(w)
    weights = np.concatenate(weights, axis=0)
    g = torch.Generator()
    g.manual_seed(int(seed))
    sampler = WeightedRandomSampler(weights, num_samples=len(concat), replacement=True, generator=g)
    return DataLoader(
        concat,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_detloc,
    )


def split_ds(ds, val_split: float, seed: int):
    n = len(ds)
    n_val = int(round(n * val_split))
    n_train = max(1, n - n_val)
    n_val = max(1, n_val)
    g = torch.Generator()
    g.manual_seed(int(seed))
    return random_split(ds, [n_train, n_val], generator=g)


def freeze_backbone(model: nn.Module, freeze: bool):
    for p in model.backbone.parameters():
        p.requires_grad = not freeze


def run_stage(
    name: str,
    model: nn.Module,
    device: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    casia_val_loader: DataLoader | None,
    cfg,
    epochs: int,
    lr: float,
    wd: float,
    freeze_epochs: int,
    ckpt_dir: str,
    log_path: str,
    save_name: str,
):
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

    ms1 = max(1, int(round(epochs * 0.6)))
    ms2 = max(ms1 + 1, int(round(epochs * 0.85)))
    scheduler = MultiStepLR(optimizer, milestones=[ms1, ms2], gamma=0.1)

    best = -1.0
    best_path = os.path.join(ckpt_dir, save_name)
    
    for ep in range(epochs):
        freeze_backbone(model, ep < freeze_epochs)

        model.train()
        run_loss = 0.0
        run_seg = 0.0
        run_clf = 0.0

        t0 = time()
        for batch in tqdm(train_loader, desc=f"{name} [{ep+1}/{epochs}]"):
            imgs = batch["image"].to(device)
            outputs = model(imgs)
            loss, seg_loss, cls_loss = multi_task_loss(
                outputs,
                batch,
                alpha=float(_get(cfg, "alpha", 10.0)),
                beta=float(_get(cfg, "beta", 1.0)),
                lambda_cls=float(_get(cfg, "lambda_cls", 0.1)),
                lambda_seg=float(_get(cfg, "lambda_seg", 1.0)),
                dice_weight=float(_get(cfg, "dice_weight", 1.0)),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run_loss += float(loss.item())
            run_seg += float(seg_loss.item())
            run_clf += float(cls_loss.item())

        scheduler.step()
        n_batches = max(1, len(train_loader))
        avg_loss = run_loss / n_batches
        avg_seg = run_seg / n_batches
        avg_clf = run_clf / n_batches
        cur_lr = scheduler.get_last_lr()[0]

        print(
            f"[{name} Ep {ep+1}] loss={avg_loss:.4f} seg={avg_seg:.4f} clf={avg_clf:.4f} lr={cur_lr:.2e} time={time()-t0:.1f}s"
        )

        val_metrics = evaluate(model, val_loader, device=device)
        print(
            f"  Val det_f1={val_metrics['det_f1']:.4f} pixel_f1={val_metrics['pixel_f1']:.4f} pixel_mcc={val_metrics['pixel_mcc']:.4f}"
        )

        casia_metrics = None
        if casia_val_loader is not None:
            casia_metrics = evaluate(model, casia_val_loader, device=device)
            print(
                f"  CASIA-Val det_f1={casia_metrics['det_f1']:.4f} pixel_f1={casia_metrics['pixel_f1']:.4f}"
            )

        with open(log_path, mode="a", newline="") as f:
            w = csv.writer(f)
            row = [
                name,
                ep + 1,
                cur_lr,
                avg_loss,
                avg_seg,
                avg_clf,
                val_metrics["det_acc"],
                val_metrics["det_prec"],
                val_metrics["det_rec"],
                val_metrics["det_f1"],
                val_metrics["det_auc"],
                val_metrics["pixel_precision"],
                val_metrics["pixel_recall"],
                val_metrics["pixel_f1"],
                val_metrics["pixel_iou"],
                val_metrics["pixel_mcc"],
            ]
            if casia_metrics is not None:
                row += [
                    casia_metrics["det_f1"],
                    casia_metrics["pixel_f1"],
                ]
            else:
                row += ["", ""]
            w.writerow(row)

        if float(val_metrics["pixel_f1"]) > best:
            best = float(val_metrics["pixel_f1"])
            state = {
                "stage": name,
                "epoch": ep + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_metric": best,
            }
            torch.save(state, best_path)
            print(f"  [Saved Best {name} -> {best_path}] (pixel_f1={best:.4f})")

    return best_path


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    if args.seed is not None:
        seed = int(args.seed)
    else:
        seed = int(_get(cfg, "seed", 42))
    set_seed(seed)

    device = _get(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.batch_size is not None:
        setattr(cfg, "batch_size", int(args.batch_size))
    if args.img_size is not None:
        setattr(cfg, "img_size", int(args.img_size))

    img_size = int(_get(cfg, "img_size", 512))
    batch_size = int(_get(cfg, "batch_size", 4))
    num_workers = int(_get(cfg, "num_workers", 4))

    ckpt_dir = _get(cfg, "ckpt_dir", "checkpoints")
    log_dir = _get(cfg, "log_dir", "logs/swintv2")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "finetune_stageB_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "stage",
                    "epoch",
                    "lr",
                    "train_loss",
                    "train_seg_loss",
                    "train_clf_loss",
                    "val_det_acc",
                    "val_det_prec",
                    "val_det_rec",
                    "val_det_f1",
                    "val_det_auc",
                    "val_pixel_precision",
                    "val_pixel_recall",
                    "val_pixel_f1",
                    "val_pixel_iou",
                    "val_pixel_mcc",
                    "casia_val_det_f1",
                    "casia_val_pixel_f1",
                ]
            )

    train_loader_casia, val_loader_casia, _, _ = build_dataloaders(cfg)
    casia_train_ds = train_loader_casia.dataset
    casia_val_loader = val_loader_casia

    coverage_ds = CoverageDetLocDataset(args.coverage_root, img_size=img_size)
    cov_train, cov_val = split_ds(coverage_ds, val_split=float(args.val_split), seed=seed)

    columbia_ds = ColumbiaDetLocDataset(args.columbia_root, img_size=img_size, dilate_k=9, include_auth=True)
    col_train, col_val = split_ds(columbia_ds, val_split=float(args.val_split), seed=seed)

    def _dl(ds):
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    cov_val_loader = DataLoader(
        cov_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_detloc
    )

    col_val_loader = DataLoader(
        col_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_detloc
    )

    

    model = SwinT2_SRM_FPN_DetLoc(
        backbone_name=_get(cfg, "backbone_name", "swinv2_tiny_window8_256"),
        pretrained_backbone=False,
        use_srm=bool(_get(cfg, "use_srm", True)),
        fpn_channels=int(_get(cfg, "fpn_channels", 256)),
        det_hidden_dim=int(_get(cfg, "det_hidden_dim", 512)),
        det_dropout=float(_get(cfg, "det_dropout", 0.30)),
        det_use_p5=bool(_get(cfg, "det_use_p5", True)),
        img_size=None,
    ).to(device)

    if not os.path.isfile(args.init_ckpt):
        raise FileNotFoundError(f"Cannot find init ckpt: {args.init_ckpt}")
    ckpt = torch.load(args.init_ckpt, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)

    casia_ratio = float(args.casia_ratio)
    casia_ratio = max(0.0, min(0.5, casia_ratio))
    other_ratio = 1.0 - casia_ratio

    cov_train_loader = make_mixed_loader(
        [cov_train, casia_train_ds],
        [other_ratio, casia_ratio],
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    b = next(iter(cov_train_loader))
    print(b["label"].shape, b["label"].dtype, b["has_mask"].shape, b["has_mask"].dtype)

    print("=== Stage 1: COVERAGE + CASIA mix ===")
    best_cov = run_stage(
        name="coverage",
        model=model,
        device=device,
        train_loader=cov_train_loader,
        val_loader=cov_val_loader,
        casia_val_loader=casia_val_loader,
        cfg=cfg,
        epochs=int(args.epochs_coverage),
        lr=float(args.lr),
        wd=float(args.weight_decay),
        freeze_epochs=int(args.freeze_backbone_epochs),
        ckpt_dir=ckpt_dir,
        log_path=log_path,
        save_name="best_swintv2_ft_coverage.pth",
    )

    ckpt_cov = torch.load(best_cov, map_location=device)
    model.load_state_dict(ckpt_cov["model"] if isinstance(ckpt_cov, dict) and "model" in ckpt_cov else ckpt_cov, strict=True)

    col_train_loader = make_mixed_loader(
        [col_train, casia_train_ds],
        [other_ratio, casia_ratio],
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed + 1,
    )

    print("=== Stage 2: Columbia + CASIA mix ===")
    best_col = run_stage(
        name="columbia",
        model=model,
        device=device,
        train_loader=col_train_loader,
        val_loader=col_val_loader,
        casia_val_loader=casia_val_loader,
        cfg=cfg,
        epochs=int(args.epochs_columbia),
        lr=float(args.lr),
        wd=float(args.weight_decay),
        freeze_epochs=int(args.freeze_backbone_epochs),
        ckpt_dir=ckpt_dir,
        log_path=log_path,
        save_name="best_swintv2_ft_columbia.pth",
    )

    ckpt_col = torch.load(best_col, map_location=device)
    model.load_state_dict(ckpt_col["model"] if isinstance(ckpt_col, dict) and "model" in ckpt_col else ckpt_col, strict=True)

    mix_ratios = [0.4, 0.4, 0.2]
    s = sum(mix_ratios)
    mix_ratios = [r / s for r in mix_ratios]

    mix_train_loader = make_mixed_loader(
        [cov_train, col_train, casia_train_ds],
        mix_ratios,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed + 2,
    )

    class _ConcatVal(Dataset):
        def __init__(self, a, b):
            self.a = a
            self.b = b
        def __len__(self):
            return len(self.a) + len(self.b)
        def __getitem__(self, i):
            if i < len(self.a):
                return self.a[i]
            return self.b[i - len(self.a)]

    mix_val_ds = _ConcatVal(cov_val, col_val)
    
    mix_val_loader = DataLoader(
            mix_val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, collate_fn=collate_detloc
        )
    print("=== Stage 3: MIX (COVERAGE + Columbia + CASIA) ===")
    best_mix = run_stage(
        name="mix",
        model=model,
        device=device,
        train_loader=mix_train_loader,
        val_loader=mix_val_loader,
        casia_val_loader=casia_val_loader,
        cfg=cfg,
        epochs=int(args.epochs_mix),
        lr=float(args.lr),
        wd=float(args.weight_decay),
        freeze_epochs=0,
        ckpt_dir=ckpt_dir,
        log_path=log_path,
        save_name="best_swintv2_ft_mix.pth",
    )

    print("Best checkpoints:")
    print("  coverage:", os.path.join(ckpt_dir, "best_swintv2_ft_coverage.pth"))
    print("  columbia:", os.path.join(ckpt_dir, "best_swintv2_ft_columbia.pth"))
    print("  mix:", os.path.join(ckpt_dir, "best_swintv2_ft_mix.pth"))


if __name__ == "__main__":
    main()

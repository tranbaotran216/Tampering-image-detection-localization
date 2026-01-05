import os
from time import time
import csv
import argparse

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from config.config import Config
from models.swint2 import SwinT2_SRM_FPN_DetLoc
from losses import multi_task_loss
from eval_utils import evaluate
from dataio.dataloaders import build_dataloaders, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train backbone swinv2 (Swin-T2)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--config", type=str, default="./config/config_swint2.yaml")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    return parser.parse_args()


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key)
    return default


def _cfg_get_path(cfg, keys, default=None):
    cur = cfg
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            cur = getattr(cur, k, None)
    return default if cur is None else cur


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    ckpt_dir = _cfg_get(cfg, "ckpt_dir", None)
    # log_dir = _cfg_get(cfg, "log_dir", None)
    log_dir = "./logs/swinv2_updated"

    if ckpt_dir is None:
        ckpt_dir = _cfg_get_path(cfg, ["checkpoints", "dir"], "checkpoints")
    if log_dir is None:
        log_dir = _cfg_get(cfg, "logs_dir", None)
    if log_dir is None:
        log_dir = _cfg_get(cfg, "logdir", None)
    if log_dir is None:
        log_dir = _cfg_get(cfg, "log_dir", "logs")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    seed = _cfg_get(cfg, "seed", 42)
    set_seed(seed)

    train_log_path = os.path.join(log_dir, "train_swint2_log.csv")
    if not os.path.exists(train_log_path):
        with open(train_log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
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
                ]
            )

    device = _cfg_get(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")

    if args.batch_size is not None:
        setattr(cfg, "batch_size", args.batch_size)

    train_loader, val_loader, _, _ = build_dataloaders(cfg)

    backbone_name = _cfg_get_path(cfg, ["model", "backbone", "model_name"], None)
    if backbone_name is None:
        backbone_name = _cfg_get(cfg, "backbone_name", "swinv2_tiny_window8_256")

    pretrained_backbone = _cfg_get_path(cfg, ["model", "backbone", "pretrained"], None)
    if pretrained_backbone is None:
        pretrained_backbone = _cfg_get(cfg, "pretrained_backbone", True)

    use_srm = _cfg_get_path(cfg, ["model", "use_srm"], None)
    if use_srm is None:
        use_srm = _cfg_get(cfg, "use_srm", True)

    fpn_channels = _cfg_get_path(cfg, ["model", "fpn_channels"], None)
    if fpn_channels is None:
        fpn_channels = _cfg_get(cfg, "fpn_channels", 256)

    det_hidden_dim = _cfg_get_path(cfg, ["model", "det", "hidden_dim"], None)
    if det_hidden_dim is None:
        det_hidden_dim = _cfg_get(cfg, "det_hidden_dim", 512)

    det_dropout = _cfg_get_path(cfg, ["model", "det", "dropout"], None)
    if det_dropout is None:
        det_dropout = _cfg_get(cfg, "det_dropout", 0.30)

    det_use_p5 = _cfg_get_path(cfg, ["model", "det", "use_p5"], None)
    if det_use_p5 is None:
        det_use_p5 = _cfg_get(cfg, "det_use_p5", True)

    model = SwinT2_SRM_FPN_DetLoc(
        backbone_name=backbone_name,
        pretrained_backbone=bool(pretrained_backbone),
        use_srm=bool(use_srm),
        fpn_channels=int(fpn_channels),
        det_hidden_dim=int(det_hidden_dim),
        det_dropout=float(det_dropout),
        det_use_p5=bool(det_use_p5),
    ).to(device)

    lr = float(_cfg_get(cfg, "learning_rate", _cfg_get_path(cfg, ["optimizer", "lr"], 1e-4)))
    wd = float(_cfg_get(cfg, "weight_decay", _cfg_get_path(cfg, ["optimizer", "weight_decay"], 0.05)))

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    milestones = _cfg_get(cfg, "lr_milestones", None)
    gamma = _cfg_get(cfg, "lr_gamma", None)
    if milestones is None:
        milestones = _cfg_get_path(cfg, ["scheduler", "milestones"], [20, 40])
    if gamma is None:
        gamma = _cfg_get_path(cfg, ["scheduler", "gamma"], 0.1)

    scheduler = MultiStepLR(optimizer, milestones=list(milestones), gamma=float(gamma))

    best_val_metric = 0.0
    best_ckpt_name = _cfg_get_path(cfg, ["checkpoints", "best"], "best_swintv2.pth")
    best_ckpt_path = os.path.join(ckpt_dir, best_ckpt_name)

    start_epoch = int(args.start_epoch)

    if args.ckpt is not None:
        if not os.path.isfile(args.ckpt):
            raise FileNotFoundError("cant find checkpoint file")

        ckpt = torch.load(args.ckpt, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            print(f"Load ckpt, optimizer, scheduler, best_val from {args.ckpt}")
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt and ckpt.get("optimizer") is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt and ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            if "best_val_metric" in ckpt:
                best_val_metric = float(ckpt["best_val_metric"])
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"])
        else:
            print("only load weights")
            model.load_state_dict(ckpt)

    num_epochs = int(_cfg_get(cfg, "num_epochs", _cfg_get_path(cfg, ["train", "epochs"], 50)))

    alpha = float(_cfg_get(cfg, "alpha", 10.0))
    beta = float(_cfg_get(cfg, "beta", 1.0))
    lambda_cls = float(_cfg_get(cfg, "lambda_cls", 0.1))
    lambda_seg = float(_cfg_get(cfg, "lambda_seg", 1.0))
    dice_weight = float(_cfg_get(cfg, "dice_weight", 1.0))

    for epoch in range(start_epoch, num_epochs):
        model.train()
        run_loss = 0.0
        run_seg = 0.0
        run_clf = 0.0

        t0 = time()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs = batch["image"].to(device)

            outputs = model(imgs)
            loss, seg_loss, cls_loss = multi_task_loss(
                outputs,
                batch,
                alpha=alpha,
                beta=beta,
                lambda_cls=lambda_cls,
                lambda_seg=lambda_seg,
                dice_weight=dice_weight,
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
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"[Epoch {epoch+1}] "
            f"loss={avg_loss:.4f} "
            f"seg={avg_seg:.4f} "
            f"clf={avg_clf:.4f} "
            f"lr={current_lr:.2e} "
            f"time={time()-t0:.1f}s"
        )

        val_metrics = evaluate(model, val_loader, device=device)
        print(
            f"  Val det_f1={val_metrics['det_f1']:.4f} "
            f"pixel_f1={val_metrics['pixel_f1']:.4f} "
            f"pixel_mcc={val_metrics['pixel_mcc']:.4f}"
        )

        with open(train_log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    current_lr,
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
            )

        if val_metrics["pixel_f1"] > best_val_metric:
            best_val_metric = float(val_metrics["pixel_f1"])
            state = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_metric": best_val_metric,
                "config_path": args.config,
            }
            torch.save(state, best_ckpt_path)
            print(f"  [Saved Best SwinV2 Model to {best_ckpt_path}] (pixel_f1={best_val_metric:.4f})")


if __name__ == "__main__":
    main()

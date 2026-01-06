import os
import numpy as np
from time import time
import csv
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from config.config import Config
from models.mobilenet import MobileNetV2_SRM_DetLoc
from losses import multi_task_loss
from eval_utils import evaluate
from datasets.dataloaders import build_dataloaders, set_seed
import argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description="Training on casia2"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the config file (yaml)",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="load ckpt to continue tranning (ex: checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="start epoch number (if load ckpt)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config) if args.config else Config()
    print("device: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    set_seed(cfg.seed)

    os.makedirs(cfg.log_dir, exist_ok=True) 
    train_log_path = os.path.join(cfg.log_dir, "train_log.csv")

    if not os.path.exists(train_log_path):
        with open(train_log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
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
            ])
    device = cfg.device
    train_loader, val_loader, _, _ = build_dataloaders(cfg)
    model = MobileNetV2_SRM_DetLoc().to(device)
    optimizer = AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    scheduler = MultiStepLR(optimizer, milestones=list(cfg.lr_milestones), gamma=cfg.lr_gamma)
    best_val_metric = 0.0
    best_ckpt_path = os.path.join(cfg.ckpt_dir, f"best_model.pth")

    start_epoch = args.start_epoch
    if args.ckpt is not None:
        if not os.path.isfile(args.ckpt):
            raise FileNotFoundError(f"cant find ckpt: {args.ckpt}")

        ckpt = torch.load(args.ckpt, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            print(f"Load ckpt from {args.ckpt}")
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt and ckpt["optimizer"] is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt and ckpt["scheduler"] is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            if "best_val_metric" in ckpt:
                best_val_metric = ckpt["best_val_metric"]
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"]
        else:
            print(f"load weights from {args.ckpt}")
            model.load_state_dict(ckpt)

        start_epoch = args.start_epoch
        
        print(f"continue training from epoch {start_epoch}")


    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        run_loss = 0.0
        run_seg = 0.0
        run_clf = 0.0

        t0 = time()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}"):
            imgs = batch["image"].to(device)          # Bx3xHxW
            # masks = batch["mask"].to(device)        # Bx1xHxW
            # labels = batch["label"].to(device)      # B
            # has_mask = batch["has_mask"].to(device) # B

            outputs = model(imgs)
            loss, seg_loss, cls_loss = multi_task_loss(
                outputs, batch,
                alpha=cfg.alpha,
                beta=cfg.beta,
                lambda_seg=cfg.lambda_seg,
                lambda_cls=cfg.lambda_cls,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            run_seg += seg_loss.item()
            run_clf += cls_loss.item()

        scheduler.step()
        n_batches = len(train_loader)
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

        # valid
        val_metrics = evaluate(model, val_loader, device=device)
        print(
            f"  Val det_f1={val_metrics['det_f1']:.4f} "
            f"pixel_f1={val_metrics['pixel_f1']:.4f} "
            f"pixel_mcc={val_metrics['pixel_mcc']:.4f}"
        )
        
        with open(train_log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
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
            ])
        if val_metrics["pixel_f1"] > best_val_metric:
            best_val_metric = val_metrics["pixel_f1"]
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best_val_metric": best_val_metric,
                "config_path": args.config,
            }
            torch.save(state, best_ckpt_path)
            print(f"  [Saved Best Model to {best_ckpt_path}] "
                  f"(pixel_f1={best_val_metric:.4f})")


if __name__ == "__main__":
    main()
import os
import csv
import torch
import numpy as np
import random
import argparse
from config import TrainConfig
from models.mobilenet import MobileNetV2_SRM_DetLoc
from eval_utils import evaluate
from datasets.dataloaders import build_dataloaders, set_seed

def parse_args():
    parser = argparse.ArgumentParser(
        description="Testing on CASIA1 and CASIA2 datasets"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--mask-th",
        type=float,
        default=0.5,
        help="Threshold for mask prediction",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = TrainConfig()

    set_seed(cfg.seed)
    device = cfg.device
    print("Device:", device)
    print("Checkpoint:", args.ckpt)
    print("Mask threshold:", args.mask_th)

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Cannot find ckpt file: {args.ckpt}")
    
    os.makedirs(cfg.log_dir, exist_ok=True)
    test_log_path = os.path.join(cfg.log_dir, "test_results.csv")
    if not os.path.exists(test_log_path):
        with open(test_log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "checkpoint",
                "dataset",
                "mask_threshold",
                "det_acc",
                "det_prec",
                "det_rec",
                "det_f1",
                "det_auc",
                "pixel_precision",
                "pixel_recall",
                "pixel_f1",
                "pixel_iou",
                "pixel_mcc",
            ])

    _, _, test_c1_loader, test_c2_loader = build_dataloaders(cfg)
    model = MobileNetV2_SRM_DetLoc().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    print("--- Testing on casia1 ---")
    metrics_c1 = evaluate(
        model, test_c1_loader, device=device, threshold=args.mask_th
    )
    for k, v in metrics_c1.items():
        print(f"{k}: {v:.4f}")

    print("--- Testing on casia2 ---")
    metrics_c2 = evaluate(
        model, test_c2_loader, device=device, threshold=args.mask_th
    )
    for k, v in metrics_c2.items():
        print(f"{k}: {v:.4f}")

    with open(test_log_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        # CASIA2_test
        writer.writerow([
            args.ckpt,
            "CASIA2_test",
            args.mask_th,
            metrics_c2["det_acc"],
            metrics_c2["det_prec"],
            metrics_c2["det_rec"],
            metrics_c2["det_f1"],
            metrics_c2["det_auc"],
            metrics_c2["pixel_precision"],
            metrics_c2["pixel_recall"],
            metrics_c2["pixel_f1"],
            metrics_c2["pixel_iou"],
            metrics_c2["pixel_mcc"],
        ])

        # CASIA1
        writer.writerow([
            args.ckpt,
            "CASIA1",
            args.mask_th,
            metrics_c1["det_acc"],
            metrics_c1["det_prec"],
            metrics_c1["det_rec"],
            metrics_c1["det_f1"],
            metrics_c1["det_auc"],
            metrics_c1["pixel_precision"],
            metrics_c1["pixel_recall"],
            metrics_c1["pixel_f1"],
            metrics_c1["pixel_iou"],
            metrics_c1["pixel_mcc"],
        ])

    print(f"\nLog test đã được ghi vào: {test_log_path}")


if __name__ == "__main__":
    main()


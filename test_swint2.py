import os
import csv
import argparse

import torch

from config.config import Config
from models.swint2 import SwinT2_SRM_FPN_DetLoc
from eval_utils import evaluate
from datasets.dataloaders import build_dataloaders, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Testing SwinV2 (Swin-T2) on CASIA1 and CASIA2 datasets")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/config_swint2.yaml",
        help="path to config yaml",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./checkpoints/best_swintv2.pth",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--mask-th",
        type=float,
        default=0.5,
        help="Threshold for mask prediction",
    )
    return parser.parse_args()


def _get(cfg, k, default=None):
    if hasattr(cfg, k):
        return getattr(cfg, k)
    if isinstance(cfg, dict):
        return cfg.get(k, default)
    return default


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    seed = _get(cfg, "seed", 42)
    set_seed(seed)

    device = _get(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Checkpoint:", args.ckpt)
    print("Mask threshold:", args.mask_th)

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Cannot find ckpt file: {args.ckpt}")

    log_dir = _get(cfg, "log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    test_log_path = ("test_results.csv")
    if not os.path.exists(test_log_path):
        with open(test_log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
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
                ]
            )

    _, _, test_c1_loader, test_c2_loader = build_dataloaders(cfg)

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

    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    print("--- Testing on CASIA1 ---")
    metrics_c1 = evaluate(model, test_c1_loader, device=device, threshold=args.mask_th)
    for k, v in metrics_c1.items():
        print(f"{k}: {v:.4f}")

    print("--- Testing on CASIA2_test ---")
    metrics_c2 = evaluate(model, test_c2_loader, device=device, threshold=args.mask_th)
    for k, v in metrics_c2.items():
        print(f"{k}: {v:.4f}")

    with open(test_log_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
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
            ]
        )

        writer.writerow(
            [
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
            ]
        )

    print(f"\nLog test đã được ghi vào: {test_log_path}")


if __name__ == "__main__":
    main()

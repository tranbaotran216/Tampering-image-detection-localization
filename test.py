import os
import csv
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.config import Config
from eval_utils import evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Unified test for Swin / ResNet50 / MobileNetV2 on CASIA / COLUMBIA / COVERAGE")

    p.add_argument("--backbone", type=str, default="swin", choices=["swin", "resnet", "mobilenet"])
    p.add_argument("--dataset", type=str, default="casia", choices=["casia", "columbia", "coverage"])

    p.add_argument("--config", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None)

    p.add_argument("--mask-th", type=float, default=0.5)

    p.add_argument("--columbia-root", type=str, default="./data/COLUMBIA")
    p.add_argument("--coverage-root", type=str, default="./data/COVERAGE")

    return p.parse_args()


def _get(cfg, k, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, k):
        return getattr(cfg, k)
    if isinstance(cfg, dict):
        return cfg.get(k, default)
    return default


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_defaults(args):
    default_cfg = {
        "swin": "./config/config_swint2.yaml",
        "resnet": "./config/config_resnet50.yaml",
        "mobilenet": "./config/config_mobilenetv2.yaml",
    }
    default_ckpt = {
        "swin": "./checkpoints/best_swintv2.pth",
        "resnet": "./checkpoints/best_resnet50.pth",
        "mobilenet": "./checkpoints/best_model_mobilenetv2.pth",
    }

    if args.config is None:
        args.config = default_cfg[args.backbone]
    if args.ckpt is None:
        args.ckpt = default_ckpt[args.backbone]

    return args


def build_model(backbone: str, cfg, device: str):
    if backbone == "swin":
        from models.swint2 import SwinT2_SRM_FPN_DetLoc

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
        return model

    if backbone == "resnet":
        from models.resnet50 import ResNet50_SRM_FPN_DetLoc

        return ResNet50_SRM_FPN_DetLoc().to(device)

    from models.mobilenet import MobileNetV2_SRM_DetLoc
    return MobileNetV2_SRM_DetLoc().to(device)


def load_ckpt(model, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)


def ensure_log(path: str):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode="w", newline="") as f:
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


def append_row(path: str, ckpt: str, ds_name: str, mask_th: float, m: dict):
    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                ckpt,
                ds_name,
                mask_th,
                m["det_acc"],
                m["det_prec"],
                m["det_rec"],
                m["det_f1"],
                m["det_auc"],
                m["pixel_precision"],
                m["pixel_recall"],
                m["pixel_f1"],
                m["pixel_iou"],
                m["pixel_mcc"],
            ]
        )


def main():
    args = parse_args()
    args = resolve_defaults(args)

    cfg = Config.from_yaml(args.config)

    seed = int(_get(cfg, "seed", 42))
    set_seed(seed)

    device = _get(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")

    print("Backbone:", args.backbone)
    print("Dataset:", args.dataset)
    print("Config:", args.config)
    print("Checkpoint:", args.ckpt)
    print("Device:", device)
    print("Mask threshold:", args.mask_th)

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Cannot find ckpt file: {args.ckpt}")

    test_log_path = "./logs/test_results.csv"
    ensure_log(test_log_path)

    model = build_model(args.backbone, cfg, device)
    load_ckpt(model, args.ckpt, device)

    num_workers = int(_get(cfg, "num_workers", 4))
    img_size = int(_get(cfg, "img_size", 512))
    dilate_k = int(_get(cfg, "dilate_k", 9))

    if args.dataset == "casia":
        from datasets.dataloaders import build_dataloaders

        _, _, test_c1_loader, test_c2_loader = build_dataloaders(cfg)

        print("--- Testing on CASIA1 ---")
        m1 = evaluate(model, test_c1_loader, device=device, threshold=args.mask_th)
        for k, v in m1.items():
            print(f"{k}: {v:.4f}")

        print("--- Testing on CASIA2_test ---")
        m2 = evaluate(model, test_c2_loader, device=device, threshold=args.mask_th)
        for k, v in m2.items():
            print(f"{k}: {v:.4f}")

        append_row(test_log_path, args.ckpt, "CASIA2_test", args.mask_th, m2)
        append_row(test_log_path, args.ckpt, "CASIA1", args.mask_th, m1)

        print(f"\nLog test đã được ghi vào: {test_log_path}")
        return

    if args.dataset == "columbia":
        from datasets.columbia import ColumbiaDetLocDataset

        ds = ColumbiaDetLocDataset(
            root=args.columbia_root,
            img_size=img_size,
            dilate_k=dilate_k,
            include_auth=True,
        )
        loader = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if str(device).startswith("cuda") else False,
        )

        print("--- Testing on COLUMBIA ---")
        m = evaluate(model, loader, device=device, threshold=args.mask_th)
        for k, v in m.items():
            print(f"{k}: {v:.4f}")

        append_row(test_log_path, args.ckpt, "COLUMBIA", args.mask_th, m)
        print(f"\nLog test đã được ghi vào: {test_log_path}")
        return

    from datasets.coverage import CoverageDetLocDataset

    ds = CoverageDetLocDataset(root=args.coverage_root, img_size=img_size)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if str(device).startswith("cuda") else False,
    )

    print("--- Testing on COVERAGE ---")
    m = evaluate(model, loader, device=device, threshold=args.mask_th)
    for k, v in m.items():
        print(f"{k}: {v:.4f}")

    append_row(test_log_path, args.ckpt, "COVERAGE", args.mask_th, m)
    print(f"\nLog test đã được ghi vào: {test_log_path}")


if __name__ == "__main__":
    main()

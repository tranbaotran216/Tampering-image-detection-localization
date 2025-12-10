# config.py
from dataclasses import dataclass, field
from typing import List, Any, Dict
import os

import torch
import yaml


@dataclass
class Config:
    # paths
    data_root: str
    casia1_name: str = "CASIA1"
    casia2_name: str = "CASIA2"

    # image
    img_size: int = 512

    # training
    batch_size: int = 8
    num_epochs: int = 100
    num_workers: int = 4

    # optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # loss
    alpha: float = 10.0
    beta: float = 1.0
    lambda_seg: float = 1.0
    lambda_cls: float = 0.1

    # scheduler
    lr_milestones: List[int] = field(default_factory=lambda: [20, 30])
    lr_gamma: float = 0.1

    # misc
    seed: int = 42
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"

    # device (tự động, có thể override nếu muốn)
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    @property
    def casia1_root(self) -> str:
        return os.path.join(self.data_root, self.casia1_name)

    @property
    def casia2_root(self) -> str:
        return os.path.join(self.data_root, self.casia2_name)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg_dict: Dict[str, Any] = yaml.safe_load(f)

        return cls(**cfg_dict)

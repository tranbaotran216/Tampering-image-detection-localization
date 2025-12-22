from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional
import os

import torch
import yaml


@dataclass
class Config:
    data_root: str
    casia1_name: str = "CASIA1"
    casia2_name: str = "CASIA2"

    img_size: int = 512

    batch_size: int = 8
    num_epochs: int = 100
    num_workers: int = 4

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    alpha: float = 10.0
    beta: float = 1.0
    lambda_seg: float = 1.0
    lambda_cls: float = 0.1

    lr_milestones: List[int] = field(default_factory=lambda: [20, 30])
    lr_gamma: float = 0.1

    seed: int = 42
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"

    model_name: str = "mobilenetv2"
    best_ckpt_name: Optional[str] = None

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    @property
    def casia1_root(self) -> str:
        return os.path.join(self.data_root, self.casia1_name)

    @property
    def casia2_root(self) -> str:
        return os.path.join(self.data_root, self.casia2_name)

    def _default_best_ckpt_name(self) -> str:
        n = (self.model_name or "").strip().lower()
        if n in {"mobilenetv2", "mobilenet", "mbv2"}:
            return "best_model_mobilenetv2.pth"
        if n in {"resnet50", "resnet", "r50"}:
            return "best_resnet50.pth"
        return f"best_{n}.pth" if n else "best_model.pth"

    @property
    def best_ckpt_path(self) -> str:
        name = self.best_ckpt_name or self._default_best_ckpt_name()
        return os.path.join(self.ckpt_dir, name)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            cfg_dict: Dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**cfg_dict)

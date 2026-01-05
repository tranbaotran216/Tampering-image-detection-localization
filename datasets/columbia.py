# datasets/columbia.py
import os
import glob
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def _is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in IMG_EXTS


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _dilate01(m01: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return m01
    pad = k // 2
    x = m01.unsqueeze(0)
    x = F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
    return (x.squeeze(0) > 0.5).float()


class ColumbiaDetLocDataset(Dataset):
    def __init__(self, root: str, img_size: int = 512, dilate_k: int = 9, include_auth: bool = True):
        super().__init__()
        self.root = root
        
        self.img_size = img_size
        self.dilate_k = int(dilate_k)
        self.include_auth = bool(include_auth)

        self.auth_dir = os.path.join(root, "4cam_auth", "4cam_auth")
        self.splc_dir = os.path.join(root, "4cam_splc", "4cam_splc")
        self.mask_dir = os.path.join(root, "4cam_splc", "edgemask")

        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])

        self.samples = []
        self._build_index()

    def _find_mask(self, stem: str) -> Optional[str]:
        cands = []
        for ext in IMG_EXTS:
            cands.extend([
                os.path.join(self.mask_dir, f"{stem}_edgemask{ext}"),
                os.path.join(self.mask_dir, f"{stem}_edgemask_3{ext}"),
            ])
        for p in cands:
            if os.path.isfile(p):
                return p
        return None

    def _build_index(self):
        if not os.path.isdir(self.splc_dir):
            raise FileNotFoundError(f"Columbia splc dir not found: {self.splc_dir}")

        if self.include_auth:
            if not os.path.isdir(self.auth_dir):
                raise FileNotFoundError(f"Columbia auth dir not found: {self.auth_dir}")
            for img_path in sorted(glob.glob(os.path.join(self.auth_dir, "*"))):
                if not _is_image_file(img_path):
                    continue
                self.samples.append((img_path, None, 0, False))

        for img_path in sorted(glob.glob(os.path.join(self.splc_dir, "*"))):
            if not _is_image_file(img_path):
                continue
            s = _stem(img_path)
            mask_path = self._find_mask(s)
            has_mask = mask_path is not None
            self.samples.append((img_path, mask_path, 1, has_mask))

        print(f"[COLUMBIA] Tổng sample: {len(self.samples)} (include_auth={self.include_auth})")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _load_mask(self, path: Optional[str], size: Tuple[int, int]) -> Image.Image:
        if path is None:
            h, w = size
            arr = np.zeros((h, w, 3), dtype=np.uint8)
            return Image.fromarray(arr, mode="RGB")
        return Image.open(path).convert("RGB")


    def __getitem__(self, idx: int):
        img_path, mask_path, label, has_mask = self.samples[idx]

        img = self._load_image(img_path)
        h, w = img.size[1], img.size[0]

        img_t = self.img_transform(img)

        if label == 0 or (not has_mask):
            mask_t = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
        else:
            mask_img = self._load_mask(mask_path, (h, w))
            m = self.mask_transform(mask_img)
            r, g, b = m[0], m[1], m[2]

            tp = (g >= (150.0 / 255.0)) & (r <= (80.0 / 255.0)) & (b <= (80.0 / 255.0))
            mask_t = tp.unsqueeze(0).float()

            if self.dilate_k > 1:
                mask_t = _dilate01(mask_t, self.dilate_k)

        return {
            "image": img_t,
            "mask": mask_t,
            "label": torch.tensor(float(label), dtype=torch.float32),
            "has_mask": torch.tensor(bool(has_mask), dtype=torch.bool),
            "path": img_path,
        }

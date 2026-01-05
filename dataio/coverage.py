# dataio/coverage.py
import os
import glob
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def _is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in IMG_EXTS


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


class CoverageDetLocDataset(Dataset):
    """
    COVERAGE dataset (copy-move/splicing)
    root/
      image/
        10.tif      (auth)
        10t.tif     (tampered)
      mask/
        10forged.tif / 10paste.tif / 10copy.tif ...

    label:
      - auth: 0
      - tampered: 1 (filename stem endswith 't')
    mask:
      - tampered: load from mask/ if exists
      - auth or missing: all-zero, has_mask=False
    """

    def __init__(self, root: str, img_size: int = 512):
        super().__init__()
        self.root = root
        self.img_size = img_size

        self.img_dir = os.path.join(root, "image")
        self.mask_dir = os.path.join(root, "mask")

        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])

        self.samples: List[Tuple[str, Optional[str], int, bool]] = []
        self._build_index()

    def _find_mask(self, base_id: str) -> Optional[str]:
        cands = []
        for ext in IMG_EXTS:
            cands.extend([
                os.path.join(self.mask_dir, f"{base_id}forged{ext}"),
                os.path.join(self.mask_dir, f"{base_id}paste{ext}"),
                os.path.join(self.mask_dir, f"{base_id}copy{ext}"),
                os.path.join(self.mask_dir, f"{base_id}{ext}"),
            ])
        for p in cands:
            if os.path.isfile(p):
                return p
        return None

    def _build_index(self):
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Coverage image dir not found: {self.img_dir}")

        for img_path in sorted(glob.glob(os.path.join(self.img_dir, "*"))):
            if not _is_image_file(img_path):
                continue
            s = _stem(img_path)
            is_t = s.endswith("t")
            base = s[:-1] if is_t else s

            if is_t:
                mask_path = self._find_mask(base)
                has_mask = mask_path is not None
                self.samples.append((img_path, mask_path, 1, has_mask))
            else:
                self.samples.append((img_path, None, 0, False))

        print(f"[COVERAGE] Tổng sample: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _load_mask(self, path: Optional[str], size: Tuple[int, int]) -> Image.Image:
        if path is None:
            h, w = size
            arr = np.zeros((h, w), dtype=np.uint8)
            return Image.fromarray(arr, mode="L")
        return Image.open(path).convert("L")

    def __getitem__(self, idx: int):
        img_path, mask_path, label, has_mask = self.samples[idx]

        img = self._load_image(img_path)
        h, w = img.size[1], img.size[0]
        mask_img = self._load_mask(mask_path, (h, w))

        img_t = self.img_transform(img)
        mask_t = self.mask_transform(mask_img)
        mask_t = (mask_t > 0.5).float()

        return {
            "image": img_t,
            "mask": mask_t,
            "label": torch.tensor(float(label), dtype=torch.float32),
            "has_mask": torch.tensor(bool(has_mask), dtype=torch.bool),  
            "path": img_path,
        }


def coverage_base_id_from_path(img_path: str) -> str:
    s = _stem(img_path)
    return s[:-1] if s.endswith("t") else s

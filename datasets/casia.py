# datasets/casia.py
import os
import glob
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def _is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in IMG_EXTS


class Casia2DetLocDataset(Dataset):
    def __init__(self, root: str, img_size: int = 512):
        super().__init__()
        self.root = root
        self.img_size = img_size

        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),                 # [0,1], 3xHxW
        ])
        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.NEAREST),
            T.ToTensor(),                 # [0,1], 1xHxW
        ])

        self.samples: List[Tuple[str, Optional[str], int, bool]] = []
        self._build_index()

    def _build_index(self):
        tp_dir = os.path.join(self.root, "Tp")
        gt_dir = os.path.join(self.root, "CASIA 2 Groundtruth")
        au_dir = os.path.join(self.root, "Au")

        # 1) Tampered + mask
        for img_path in sorted(glob.glob(os.path.join(tp_dir, "*"))):
            if not _is_image_file(img_path):
                continue
            base = os.path.splitext(os.path.basename(img_path))[0]

            mask_path = None
            for ext in IMG_EXTS:
                cand = os.path.join(gt_dir, base + "_gt" + ext)
                if os.path.exists(cand):
                    mask_path = cand
                    break

            if mask_path is None:
                continue

            # (img, mask, label=1, has_mask=True)
            self.samples.append((img_path, mask_path, 1, True))

        # 2) Authentic (label=0, không mask)
        for img_path in sorted(glob.glob(os.path.join(au_dir, "*"))):
            if not _is_image_file(img_path):
                continue
            self.samples.append((img_path, None, 0, False))

        print(f"[CASIA2] Tổng sample: {len(self.samples)} "
              f"(Tp có mask + Au không mask)")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def _load_mask(self, path: Optional[str], size: Tuple[int, int]) -> Image.Image:
        if path is None:
            # mask all-zero
            h, w = size
            arr = np.zeros((h, w), dtype=np.uint8)
            return Image.fromarray(arr, mode="L")

        m = Image.open(path).convert("L")
        return m

    def __getitem__(self, idx: int):
        img_path, mask_path, label, has_mask = self.samples[idx]

        img = self._load_image(img_path)
        h, w = img.size[1], img.size[0]

        mask_img = self._load_mask(mask_path, (h, w))

        img_t = self.img_transform(img)          # 3x512x512
        mask_t = self.mask_transform(mask_img)   # 1x512x512, 0..1

        # Binarize mask (nhiều mask là 0/255)
        mask_t = (mask_t > 0.5).float()

        label_t = torch.tensor(label, dtype=torch.long)
        has_mask_t = torch.tensor(has_mask, dtype=torch.bool)

        return {
            "image": img_t,
            "mask": mask_t,
            "label": label_t,
            "has_mask": has_mask_t,
            "path": img_path,
        }
class Casia1DetDataset(Dataset):
    """
    CASIA v1.0
    root/
      Au/
      Sp/

    - Au: label=0
    - Sp: label=1
    Không dùng mask (chỉ detection). Mask all-zero, has_mask=False.
    """

    def __init__(self, root: str, img_size: int = 512):
        super().__init__()
        self.root = root
        self.img_size = img_size

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

    def _build_index(self):
        au_dir = os.path.join(self.root, "Au")
        sp_dir = os.path.join(self.root, "Sp")

        # Authentic
        for img_path in sorted(glob.glob(os.path.join(au_dir, "*"))):
            if not _is_image_file(img_path):
                continue
            self.samples.append((img_path, None, 0, False))

        # Spliced / Tampered
        for img_path in sorted(glob.glob(os.path.join(sp_dir, "*"))):
            if not _is_image_file(img_path):
                continue
            self.samples.append((img_path, None, 1, False))

        print(f"[CASIA1] Tổng sample: {len(self.samples)} (Au + Sp, không mask)")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int):
        img_path, _, label, has_mask = self.samples[idx]
        img = self._load_image(img_path)

        img_t = self.img_transform(img)
        # mask all-zero
        mask_img = Image.fromarray(
            np.zeros((img.size[1], img.size[0]), dtype=np.uint8),
            mode="L",
        )
        mask_t = self.mask_transform(mask_img)
        mask_t = (mask_t > 0.5).float()

        label_t = torch.tensor(label, dtype=torch.long)
        has_mask_t = torch.tensor(has_mask, dtype=torch.bool)

        return {
            "image": img_t,
            "mask": mask_t,
            "label": label_t,
            "has_mask": has_mask_t,
            "path": img_path,
        }

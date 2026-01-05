# swint2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Optional
import timm


class SRMConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("weight", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        weight = self.weight.repeat(c, 1, 1, 1)
        y = F.conv2d(x, weight, bias=None, stride=1, padding=1, groups=c)
        return y

class SRMGray(nn.Module):
    def __init__(self, out_channels_list = [64, 128, 256, 256]):
        super().__init__()
        c2, c3,c4,c5 = out_channels_list

        self.s1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.s2 = nn.Sequential(
            nn.Conv2d(32, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

        self.s3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.s4 = nn.Sequential(
            nn.Conv2d(c3, c4, 3,2,1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

        self.s5 = nn.Sequential(
            nn.Conv2d(c4, c5, 3,2,1),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        x = self.s1(x) # b,32,h/2,w/2
        g2 = self.s2(x) # b,c2,h/4,w/4
        g3 = self.s3(g2)# b,c3,h/8,w/8
        g4 = self.s4(g3)# b,c4,h/16,w/16
        g5 = self.s5(g4)# b,c5,h/32,w/32
        return g2, g3, g4, g5


class SwinV2Backbone(nn.Module):
    def __init__(
        self,
        model_name: str = "swinv2_tiny_window8_256",
        pretrained: bool = True,
        out_indices=(0, 1, 2, 3),
        img_size:Optional[int] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.out_indices = out_indices
        self.img_size = img_size

        self._backend = None
        self._features_only = None
        self._feature_info = None
        self._channels = None

        self._build_backbone()

    def _disable_strict_img_size(self):
        fo = self._features_only
        pe = getattr(fo, "patch_embed", None)
        if pe is None:
            base = getattr(fo, "model", None)
            pe = getattr(base, "patch_embed", None) if base is not None else None
        if pe is not None and hasattr(pe, "strict_img_size"):
            pe.strict_img_size = False

    def _maybe_set_input_size(self, h: int, w: int):
        fo = self._features_only
        fn = getattr(fo, "set_input_size", None)
        if callable(fn):
            try:
                fn((h, w))
            except Exception:
                pass

    def _build_backbone(self):
        try:
            self._backend = "timm"
            kwargs = dict(
                pretrained=self.pretrained,
                features_only=True,
                out_indices=self.out_indices,
            )
            if self.img_size is not None:
                kwargs["img_size"] = int(self.img_size)

            self._features_only = timm.create_model(self.model_name, **kwargs)
            self._feature_info = self._features_only.feature_info
            self._disable_strict_img_size()
        except Exception:
            try:
                self._backend = "torchvision"

                if self.model_name in ("swin_v2_t", "swinv2_t"):
                    m = swin_v2_t(weights="DEFAULT" if self.pretrained else None)
                    return_nodes = {
                        "features.1": "s1",
                        "features.3": "s2",
                        "features.5": "s3",
                        "features.7": "s4",
                    }
                    self._channels = [96, 192, 384, 768]
                elif self.model_name in ("swin_v2_s", "swinv2_s"):
                    m = swin_v2_s(weights="DEFAULT" if self.pretrained else None)
                    return_nodes = {
                        "features.1": "s1",
                        "features.3": "s2",
                        "features.5": "s3",
                        "features.7": "s4",
                    }
                    self._channels = [96, 192, 384, 768]
                elif self.model_name in ("swin_v2_b", "swinv2_b"):
                    m = swin_v2_b(weights="DEFAULT" if self.pretrained else None)
                    return_nodes = {
                        "features.1": "s1",
                        "features.3": "s2",
                        "features.5": "s3",
                        "features.7": "s4",
                    }
                    self._channels = [128, 256, 512, 1024]
                else:
                    raise ValueError(f"Unknown torchvision SwinV2 name: {self.model_name}")

                self._features_only = create_feature_extractor(m, return_nodes=return_nodes)
                self._feature_info = None
            except Exception as e:
                raise RuntimeError(
                    "Cannot build SwinV2 backbone. Install timm or use a working torchvision build."
                ) from e

    def channels(self):
        if self._backend == "timm":
            return list(self._feature_info.channels())
        return list(self._channels)

    def forward(self, x: torch.Tensor):
        if self._backend == "timm":
            h, w = x.shape[-2], x.shape[-1]
            self._disable_strict_img_size()
            self._maybe_set_input_size(h, w)

            feats = self._features_only(x)

            exp_ch = list(self._feature_info.channels()) if self._feature_info is not None else None
            fixed = []
            for i, f in enumerate(feats):
                if f.dim() == 4:
                    if exp_ch is not None and i < len(exp_ch):
                        c = exp_ch[i]
                        if f.shape[1] != c and f.shape[-1] == c:
                            f = f.permute(0, 3, 1, 2).contiguous()
                    else:
                        if f.shape[1] not in (32, 48, 64, 96, 128, 160, 192, 256, 384, 512, 768, 1024) and f.shape[-1] in (
                            32, 48, 64, 96, 128, 160, 192, 256, 384, 512, 768, 1024
                        ):
                            f = f.permute(0, 3, 1, 2).contiguous()
                fixed.append(f)

            return fixed[0], fixed[1], fixed[2], fixed[3]


        out = self._features_only(x)
        feats = [out["s1"], out["s2"], out["s3"], out["s4"]]
        feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
        return feats[0], feats[1], feats[2], feats[3]


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list]
        )
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list]
        )

    def forward(self, c2, c3, c4, c5):
        feats = [c2, c3, c4, c5]
        laterals = [l(f) for l, f in zip(self.lateral_convs, feats)]

        p5 = laterals[3]
        p4 = laterals[2] + F.interpolate(p5, size=laterals[2].shape[2:], mode="nearest")
        p3 = laterals[1] + F.interpolate(p4, size=laterals[1].shape[2:], mode="nearest")
        p2 = laterals[0] + F.interpolate(p3, size=laterals[0].shape[2:], mode="nearest")

        p2 = self.output_convs[0](p2)
        p3 = self.output_convs[1](p3)
        p4 = self.output_convs[2](p4)
        p5 = self.output_convs[3](p5)
        return p2, p3, p4, p5


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor, out_size):
        x = F.relu(self.conv1(feat), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        mask_logits = self.out_conv(x)
        mask_logits = F.interpolate(mask_logits, size=out_size, mode="bilinear", align_corners=False)
        mask_prob = torch.sigmoid(mask_logits)
        return mask_logits, mask_prob


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 512, dropout: float = 0.30):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, feat: torch.Tensor):
        x = self.pool(feat).flatten(1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.drop(x)
        logits = self.fc2(x)
        prob = torch.sigmoid(logits)
        return logits, prob


class SwinT2_SRM_FPN_DetLoc(nn.Module):
    def __init__(
        self,
        backbone_name: str = "swinv2_tiny_window8_256",
        pretrained_backbone: bool = True,
        use_srm: bool = True,
        fpn_channels: int = 256,
        det_hidden_dim: int = 512,
        det_dropout: float = 0.30,
        det_use_p5: bool = True,
        img_size: Optional[int] = None,
        srm_out_channels = (64, 128, 256, 256),
        srm_gamma_init: float = 0.05,
        
    ):
        super().__init__()
        self.use_srm = use_srm
        self.det_use_p5 = det_use_p5

        self.srm = SRMConv2d() if use_srm else None

        self.backbone = SwinV2Backbone(
            model_name=backbone_name,
            pretrained=pretrained_backbone,
            img_size=img_size,
        )
        in_ch = self.backbone.channels()
        self.fpn = FPN(in_ch, out_channels=fpn_channels)

        self.seg_head = SegmentationHead(in_channels=fpn_channels)
        det_in = fpn_channels if det_use_p5 else in_ch[-1]
        self.det_head = DetectionHead(in_channels=det_in, hidden_dim=det_hidden_dim, dropout=det_dropout)

        self.gray_srm = SRMGray(out_channels_list = srm_out_channels) if use_srm else None
        c2,c3,c4,c5 = srm_out_channels
        self.gray_proj = nn.ModuleList([
            nn.Conv2d(c2, fpn_channels, kernel_size=1),
            nn.Conv2d(c3, fpn_channels, kernel_size=1),
            nn.Conv2d(c4, fpn_channels, kernel_size=1), 
            nn.Conv2d(c5, fpn_channels, kernel_size=1),
        ])
        self.srm_gamma = nn.Parameter(torch.tensor([srm_gamma_init] * 4, dtype=torch.float32))


    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]

        c2, c3, c4, c5 = self.backbone(x)
        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)

        if self.use_srm and self.srm is not None:
            gray = x.mean(dim=1, keepdim=True)  # Bx1xHxW
            srm1 = self.srm(gray)  # Bx1xHxW
            g2, g3, g4, g5 = self.gray_srm(srm1)
            p2 = p2 + self.srm_gamma[0] * self.gray_proj[0](g2)
            p3 = p3 + self.srm_gamma[1] * self.gray_proj[1](g3)
            p4 = p4 + self.srm_gamma[2] * self.gray_proj[2](g4)
            p5 = p5 + self.srm_gamma[3] * self.gray_proj[3](g5)
            
        mask_logits, mask_prob = self.seg_head(p2, input_size)

        det_feat = p5 if self.det_use_p5 else c5
        det_logits, det_prob = self.det_head(det_feat)

        return {
            "det_logits": det_logits,
            "det_prob": det_prob,
            "mask_logits": mask_logits,
            "mask_prob": mask_prob,
        }


__all__ = [
    "SRMConv2d",
    "SwinV2Backbone",
    "FPN",
    "SegmentationHead",
    "DetectionHead",
    "SwinT2_SRM_FPN_DetLoc",
]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SwinT2_SRM_FPN_DetLoc(
        backbone_name="swinv2_tiny_window8_256",
        pretrained_backbone=False,
        use_srm=True,
        det_use_p5=True,
        img_size=None,
    ).to(device)

    x1 = torch.randn(1, 3, 512, 512, device=device)
    out1 = model(x1)
    print("512 mask:", out1["mask_prob"].shape, "det:", out1["det_prob"].shape)

    x2 = torch.randn(1, 3, 256, 256, device=device)
    out2 = model(x2)
    print("256 mask:", out2["mask_prob"].shape, "det:", out2["det_prob"].shape)

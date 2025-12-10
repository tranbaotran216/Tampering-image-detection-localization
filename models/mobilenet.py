# models/mobilenetv2_srm_detloc.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ----------------------- SRM FILTER ----------------------- #
class SRMConv2d(nn.Module):
    """
    Lọc SRM: xám Bx1xHxW -> Bx3xHxW (3 kernel high-pass 5x5).
    """
    def __init__(self):
        super().__init__()
        # 3 kernel SRM
        k5 = torch.tensor([
            [0, 0,  0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0,  0, 0, 0],
        ], dtype=torch.float32)

        k3 = torch.tensor([
            [-1, 2, -1],
            [2, -4, 2],
            [-1, 2, -1],
        ], dtype=torch.float32)

        k13 = torch.tensor([[-1, 2, -1]], dtype=torch.float32)

        # pad lên 5x5
        k3_5 = torch.zeros((5, 5), dtype=torch.float32)
        k3_5[1:4, 1:4] = k3

        k13_5 = torch.zeros((5, 5), dtype=torch.float32)
        k13_5[2, 1:4] = k13

        weight = torch.stack([k5, k3_5, k13_5], dim=0)  # (3,5,5)
        weight = weight.unsqueeze(1)                    # (3,1,5,5)

        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        # x: Bx1xHxW
        return F.conv2d(x, self.weight, padding=2)


# ----------------------- MOBILENETV2 BACKBONE ----------------------- #
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = (self.stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2Backbone(nn.Module):
    """
    Backbone MobileNetV2 cho 512x512.
    Trả ra:
      - shallow: H/4 x W/4 (128x128)  (sau stage 2, C=24)
      - deep   : H/16 x W/16 (32x32)  (cuối cùng, C=320)
    """
    def __init__(self, in_channels=3):
        super().__init__()
        input_channel = 32

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, input_channel, kernel_size=3,
                      stride=2, padding=1, bias=False),  # 512->256
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        )

        # (t, c, n, s)
        cfgs = [
            [1,  16, 1, 1],  # stage1 256x256
            [6,  24, 2, 2],  # stage2 128x128 (shallow)
            [6,  32, 3, 2],  # stage3 64x64
            [6,  64, 4, 2],  # stage4 32x32
            [6,  96, 3, 1],  # stage5 32x32
            [6, 160, 3, 1],  # stage6 32x32
            [6, 320, 1, 1],  # stage7 32x32
        ]

        def make_stage(c_in, t, c_out, n, s):
            blocks = []
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(InvertedResidual(c_in, c_out, stride, t))
                c_in = c_out
            return nn.Sequential(*blocks), c_in

        c = input_channel
        self.stage1, c = make_stage(c, *cfgs[0])  # ->16, 256x256
        self.stage2, c = make_stage(c, *cfgs[1])  # ->24, 128x128
        self.stage3, c = make_stage(c, *cfgs[2])  # ->32, 64x64
        self.stage4, c = make_stage(c, *cfgs[3])  # ->64, 32x32
        self.stage5, c = make_stage(c, *cfgs[4])  # ->96, 32x32
        self.stage6, c = make_stage(c, *cfgs[5])  # ->160,32x32
        self.stage7, c = make_stage(c, *cfgs[6])  # ->320,32x32

        self.out_channels = c           # 320
        self.shallow_out_channels = 24  # sau stage2

    def forward(self, x):
        # x: Bx3x512x512
        x = self.first_conv(x)   # Bx32x256x256
        x = self.stage1(x)       # Bx16x256x256
        x = self.stage2(x)       # Bx24x128x128
        shallow = x
        x = self.stage3(x)       # Bx32x64x64
        x = self.stage4(x)       # Bx64x32x32
        x = self.stage5(x)       # Bx96x32x32
        x = self.stage6(x)       # Bx160x32x32
        x = self.stage7(x)       # Bx320x32x32
        deep = x
        return shallow, deep


# ----------------------- CBAM ----------------------- #
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(B, C)
        max_pool = F.adaptive_max_pool2d(x, 1).view(B, C)
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        scale = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg, maxv], dim=1)   # Bx2xHxW
        attn = torch.sigmoid(self.conv(attn))
        return x * attn


class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ----------------------- DUAL-STREAM BACKBONE ----------------------- #
class DualStreamBackbone(nn.Module):
    """
    2 stream:
      - RGB: MobileNetV2Backbone(3)
      - SRM: Gray -> SRMConv2d -> MobileNetV2Backbone(3)
    Trả ra:
      - deep_fused: B x deep_out_ch x 32 x 32
      - shallow_fused: B x (2*shallow_out_ch) x 128 x 128
    """
    def __init__(self,
                 rgb_in_ch=3,
                 srm_in_ch=3,
                 deep_out_ch=128,
                 shallow_out_ch=64):
        super().__init__()
        self.rgb_backbone = MobileNetV2Backbone(in_channels=rgb_in_ch)
        self.srm_backbone = MobileNetV2Backbone(in_channels=srm_in_ch)
        self.srm_conv = SRMConv2d()

        self.deep_out_ch = deep_out_ch
        self.shallow_out_ch = shallow_out_ch

        # proj deep 320 -> deep_out_ch
        self.rgb_deep_proj = nn.Conv2d(self.rgb_backbone.out_channels,
                                       deep_out_ch, 1)
        self.srm_deep_proj = nn.Conv2d(self.srm_backbone.out_channels,
                                       deep_out_ch, 1)

        # proj shallow 24 -> shallow_out_ch
        self.rgb_shallow_proj = nn.Conv2d(
            self.rgb_backbone.shallow_out_channels,
            shallow_out_ch, 1)
        self.srm_shallow_proj = nn.Conv2d(
            self.srm_backbone.shallow_out_channels,
            shallow_out_ch, 1)

        # fusion deep: concat(2*deep_out_ch) -> deep_out_ch
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(deep_out_ch * 2, deep_out_ch,
                      kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(deep_out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(deep_out_ch, deep_out_ch,
                      kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(deep_out_ch),
            nn.ReLU(inplace=True),
        )
        self.cbam_deep = CBAMBlock(deep_out_ch)

    def forward(self, x):
        # RGB stream
        rgb_shallow, rgb_deep = self.rgb_backbone(x)

        # SRM stream
        gray = x.mean(dim=1, keepdim=True)
        srm = self.srm_conv(gray)
        srm_shallow, srm_deep = self.srm_backbone(srm)

        # deep proj + fusion
        rgb_deep_p = self.rgb_deep_proj(rgb_deep)
        srm_deep_p = self.srm_deep_proj(srm_deep)
        deep_cat = torch.cat([rgb_deep_p, srm_deep_p], dim=1)
        deep = self.dilated_conv(deep_cat)
        deep = self.cbam_deep(deep)  # B x deep_out_ch x 32 x 32

        # shallow proj + concat
        rgb_shallow_p = self.rgb_shallow_proj(rgb_shallow)
        srm_shallow_p = self.srm_shallow_proj(srm_shallow)
        shallow = torch.cat([rgb_shallow_p, srm_shallow_p], dim=1)
        # B x (2*shallow_out_ch) x 128 x 128

        return deep, shallow


# ----------------------- HEAD LOCALIZATION ----------------------- #
class UNetLiteHead(nn.Module):
    def __init__(self, in_channels_deep, in_channels_shallow,
                 mid_channels=128, seg_channels=64):
        super().__init__()
        self.conv_deep = nn.Conv2d(in_channels_deep, mid_channels, 1)
        self.conv_shallow = nn.Conv2d(in_channels_shallow, mid_channels, 1)

        fusion_in = mid_channels * 2

        self.conv_block_128 = nn.Sequential(
            nn.Conv2d(fusion_in, seg_channels, 3, padding=1),
            nn.BatchNorm2d(seg_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(seg_channels, seg_channels, 3, padding=1),
            nn.BatchNorm2d(seg_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_out = nn.Conv2d(seg_channels, 1, kernel_size=1)

    def forward(self, deep_fused, shallow_fused):
        # deep_fused: B x C_d x 32 x 32
        # shallow_fused: B x C_s x 128 x 128
        deep = self.conv_deep(deep_fused)
        shallow = self.conv_shallow(shallow_fused)

        _, _, Hs, Ws = shallow.shape
        deep_up = F.interpolate(deep, size=(Hs, Ws),
                                mode="bilinear", align_corners=False)

        x = torch.cat([deep_up, shallow], dim=1)
        x = self.conv_block_128(x)

        mask_logits = self.conv_out(x)  # Bx1x128x128
        mask_logits = F.interpolate(mask_logits, scale_factor=4,
                                    mode="bilinear", align_corners=False)
        mask_prob = torch.sigmoid(mask_logits)
        return mask_prob, mask_logits


# ----------------------- HEAD DETECTION ----------------------- #
class DetectionHead(nn.Module):
    def __init__(self, in_channels_deep, hidden_dim=128, use_maxpool=True):
        super().__init__()
        self.use_maxpool = use_maxpool
        if use_maxpool:
            mlp_in = in_channels_deep * 2
        else:
            mlp_in = in_channels_deep

        self.fc1 = nn.Linear(mlp_in, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, deep_fused):
        B, C, H, W = deep_fused.shape
        gap = F.adaptive_avg_pool2d(deep_fused, 1).view(B, C)
        if self.use_maxpool:
            gmp = F.adaptive_max_pool2d(deep_fused, 1).view(B, C)
            feat = torch.cat([gap, gmp], dim=1)
        else:
            feat = gap

        x = self.fc1(feat)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        logits = self.fc2(x).view(-1)
        prob = torch.sigmoid(logits)
        return prob, logits


# ----------------------- FULL MODEL ----------------------- #
class MobileNetV2_SRM_DetLoc(nn.Module):
    def __init__(self,
                 deep_out_ch=128,
                 shallow_out_ch=64,
                 mid_channels=128,
                 seg_channels=64,
                 det_hidden_dim=128):
        super().__init__()
        self.backbone = DualStreamBackbone(
            rgb_in_ch=3,
            srm_in_ch=3,
            deep_out_ch=deep_out_ch,
            shallow_out_ch=shallow_out_ch,
        )

        self.loc_head = UNetLiteHead(
            in_channels_deep=deep_out_ch,
            in_channels_shallow=shallow_out_ch * 2,
            mid_channels=mid_channels,
            seg_channels=seg_channels,
        )

        self.det_head = DetectionHead(
            in_channels_deep=deep_out_ch,
            hidden_dim=det_hidden_dim,
            use_maxpool=True,
        )

    def forward(self, x):
        deep, shallow = self.backbone(x)
        mask_prob, mask_logits = self.loc_head(deep, shallow)
        det_prob, det_logits = self.det_head(deep)
        return {
            "mask_prob": mask_prob,
            "mask_logits": mask_logits,
            "det_prob": det_prob,
            "det_logits": det_logits,
        }


if __name__ == "__main__":
    # test nhanh
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MobileNetV2_SRM_DetLoc().to(device)
    x = torch.randn(1, 3, 512, 512, device=device)
    out = model(x)
    print("mask_prob:", out["mask_prob"].shape)
    print("det_prob:", out["det_prob"].shape)

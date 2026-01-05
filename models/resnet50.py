import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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
        ).view(1,1,3,3)
        # 3 channels
        weight = kernel.repeat(3,1,1,1)  # 3x1x3x3

        self.conv=nn.Conv2d(
            in_channels=3,
            out_channels = 3,
            kernel_size=3,
            padding=1,
            bias = False,
            groups=3,
        )
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        for p in self.conv.parameters():
            p.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
        
class ResNet50Backbone(nn.Module):
    """
    fixed img size: 3 x 512 x 512
    C2: 256 x h/4 x w/4
    C3: 512 x h/8 x w/8
    C4: 1024 x h/16 x w/16
    C5: 2048 x h/32 x w/32
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        except AttributeError:
            backbone = models.resnet50(pretrained=pretrained)

        self.conv1 = backbone.conv1  # 64 x 256 x 256
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # C2: 256 x 128 x 128
        self.layer2 = backbone.layer2  # C3: 512 x 64 x 64
        self.layer3 = backbone.layer3  # C4: 1024 x 32 x 32
        self.layer4 = backbone.layer4  # C5: 2048 x 16 x 16

    def forward(self, x: torch.Tensor):
        # x: b, 3, 512, 512
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # b, 64, 128, 128

        c2 = self.layer1(x) # b, 256, 128, 128
        c3 = self.layer2(c2) # b, 512, 64, 64
        c4 = self.layer3(c3) # b, 1024, 32, 32
        c5 = self.layer4(c4) # b, 2048, 16, 16
        return c2, c3, c4, c5
    
class FPN(nn.Module):
    def __init__(self, in_channels = (256, 512, 1024, 2048), out_channels: int = 256):
        super().__init__()
        c2_in, c3_in, c4_in, c5_in = in_channels
        # lateral convs 1x1
        self.lat2 = nn.Conv2d(c2_in, out_channels, kernel_size=1)
        self.lat3 = nn.Conv2d(c3_in, out_channels, kernel_size=1)
        self.lat4 = nn.Conv2d(c4_in, out_channels, kernel_size=1)
        self.lat5 = nn.Conv2d(c5_in, out_channels, kernel_size=1)

        # smooth convs 3x3
        self.smooth2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def _upsample_add(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        # up x -> add to y
        x_ups = F.interpolate(x, size=y.shape[2:], mode="bilinear", align_corners=False)
        return x_ups + y
    
    def forward(self, c2, c3,c4,c5):
        m5 = self.lat5(c5)  # b, 256, h/32, w/32
        m4 = self.lat4(c4)  # b, 256, h/16, w/16
        m3 = self.lat3(c3)  # b, 256, h/8, w/8
        m2 = self.lat2(c2)  # b, 256, h/4, w/4

        p5 = self.smooth5(m5)
        m4 = self._upsample_add(m5, m4)
        p4 = self.smooth4(m4)
        m3 = self._upsample_add(m4, m3)
        p3 = self.smooth3(m3)
        m2 = self._upsample_add(m3, m2)
        p2 = self.smooth2(m2)
        return p2, p3, p4, p5
    
class SegmentationHead(nn.Module):
    # input: p2 [b, 256, h/4, w/4]
    # conv 3x3: c = 128 -> conv 3x3: c=64 -> conv1x1 -> c=1 (mask_logits) -> upsample 512x512
    def __init__(self, in_channels: int = 256, mid_channels:int=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        out_channels = mid_channels//2
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv_out = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, p2: torch.Tensor, input_size):
        #input size: hxw (512, 512)
        x = self.conv1(p2)
        x = self.bn1(x)
        x = F.relu(x,  inplace=True)

        x = self.conv2(x)           # B x 64 x H/4 x W/4
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        
        mask_logits_low = self.conv_out(x) #b,1,128, 128

        mask_logits = F.interpolate(
            mask_logits_low,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        ) #b,1,512,512

        mask_prob = torch.sigmoid(mask_logits)
        return mask_logits, mask_prob
    
class DetectionHead(nn.Module):
    #input: p5 (b, 2048, 16,16)
    # adaptiveAvgPool -> fc -> sigmoid
    def __init__(self, in_channels: int = 2048, mid_channels:int=512, num_classes:int=1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.dropout= nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(mid_channels, num_classes)

    def forward(self, c5: torch.Tensor):
        x = self.pool(c5) #b, 2048, 1, 1
        x = x.flatten(1) # b, 2048
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        logits = self.fc2(x) 
        prob = torch.sigmoid(logits)
        return logits, prob
        
class ResNet50_SRM_FPN_DetLoc(nn.Module):
    def __init__(
        self,
        pretrained_backbone: bool = True,
        use_srm: bool = True,
        fpn_channels: int = 256,
    ):
        super().__init__()
        self.use_srm = use_srm
        if use_srm:
            self.srm = SRMConv2d()
        else:
            self.srm = None

        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)

        self.fpn = FPN(
            in_channels=(256, 512, 1024, 2048),
            out_channels=fpn_channels,
        )

        self.seg_head = SegmentationHead(
            in_channels=fpn_channels,
            mid_channels=128,
        )

        self.det_head = DetectionHead(
            in_channels=2048,
            mid_channels=512,
            num_classes=1,
        )

    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]  # (H, W)

        if self.use_srm and self.srm is not None:
            x = x + self.srm(x)

        c2, c3, c4, c5 = self.backbone(x)     # deep & shallow features
        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)

        mask_logits, mask_prob = self.seg_head(p2, input_size)
        det_logits, det_prob = self.det_head(c5)

        return {
            "det_logits": det_logits,
            "det_prob": det_prob,
            "mask_logits": mask_logits,
            "mask_prob": mask_prob,
        }


__all__ = [
    "SRMConv2d",
    "ResNet50Backbone",
    "FPN",
    "SegmentationHead",
    "DetectionHead",
    "ResNet50_SRM_FPN_DetLoc",
]
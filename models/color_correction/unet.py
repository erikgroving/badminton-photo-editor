"""
Color correction U-Net: ResNet-34 encoder + decoder with skip connections.

Learns the residual (edited - raw), which is then added back to produce the
corrected image. Loss = pixel L1 + perceptual (VGG features) + judge adversarial.

The adversarial component uses the frozen judge model to steer outputs toward
the distribution of Jay's edited photos.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = _ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ColorUNet(nn.Module):
    """
    Encoder: ResNet-34 pretrained backbone.
    Decoder: 4 upsampling stages with skip connections.
    Output: 3-channel residual in [-1, 1] (add to input to get corrected image).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm
        enc = tvm.resnet34(weights="IMAGENET1K_V1" if pretrained else None)

        self.enc0 = nn.Sequential(enc.conv1, enc.bn1, enc.relu)   # 64ch, /2
        self.pool = enc.maxpool
        self.enc1 = enc.layer1   # 64ch,  /4
        self.enc2 = enc.layer2   # 128ch, /8
        self.enc3 = enc.layer3   # 256ch, /16
        self.enc4 = enc.layer4   # 512ch, /32

        self.bottleneck = _ConvBlock(512, 512)

        self.up4 = _UpBlock(512, 256, 256)
        self.up3 = _UpBlock(256, 128, 128)
        self.up2 = _UpBlock(128, 64,   64)
        self.up1 = _UpBlock(64,  64,   64)

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.head     = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        e0 = self.enc0(x)         # /2
        ep = self.pool(e0)
        e1 = self.enc1(ep)        # /4
        e2 = self.enc2(e1)        # /8
        e3 = self.enc3(e2)        # /16
        e4 = self.enc4(e3)        # /32

        b  = self.bottleneck(e4)

        d  = self.up4(b, e3)
        d  = self.up3(d, e2)
        d  = self.up2(d, e1)
        d  = self.up1(d, e0)
        d  = self.final_up(d)
        if d.shape[2:] != x.shape[2:]:
            d = F.interpolate(d, size=x.shape[2:], mode="bilinear", align_corners=False)
        residual = self.head(d)
        return torch.clamp(x + residual, 0.0, 1.0)


# ── Perceptual loss helper ────────────────────────────────────────────────────

class _VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        vgg = tvm.vgg16(weights="IMAGENET1K_V1").features
        self.slice = nn.Sequential(*list(vgg.children())[:16])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.slice(pred), self.slice(target))


class UNetLoss(nn.Module):
    def __init__(self, lambda_pixel: float, lambda_percept: float, lambda_judge: float,
                 judge_model: nn.Module | None = None):
        super().__init__()
        self.lambda_pixel   = lambda_pixel
        self.lambda_percept = lambda_percept
        self.lambda_judge   = lambda_judge
        self.perceptual     = _VGGPerceptualLoss() if lambda_percept > 0 else None
        self.judge          = judge_model  # frozen

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        pixel_loss   = F.l1_loss(pred, target)
        percept_loss = self.perceptual(pred, target) if self.perceptual else torch.zeros(1, device=pred.device)
        judge_loss   = torch.zeros(1, device=pred.device)
        if self.judge is not None and self.lambda_judge > 0:
            # Maximise judge score (wants pred to look "edited")
            logits     = self.judge(pred).squeeze(1)
            judge_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))

        total = (self.lambda_pixel   * pixel_loss
               + self.lambda_percept * percept_loss
               + self.lambda_judge   * judge_loss)
        return total, {
            "pixel":   pixel_loss.item(),
            "percept": percept_loss.item(),
            "judge":   judge_loss.item(),
        }

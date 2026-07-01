"""
Judge / discriminator model: any timm backbone trained to distinguish
raw (0) vs fully-edited (1) images.

Used as:
  1. A standalone quality scorer (frozen, for UI)
  2. An adversarial discriminator during GAN-style color correction training
     (unfrozen, jointly trained with the color model)
"""
import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import JUDGE_BACKBONE_CANDIDATES, JUDGE_MODEL_NAME


def build_model(backbone: str = JUDGE_MODEL_NAME, pretrained: bool = True) -> nn.Module:
    import timm
    return timm.create_model(backbone, pretrained=pretrained, num_classes=1)

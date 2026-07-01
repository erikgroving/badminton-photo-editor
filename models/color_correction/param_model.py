"""
Color correction parameter model: EfficientNet-B4 → 9 Lightroom-style sliders.

Predicts [exposure, contrast, highlights, shadows, whites, blacks,
          temp_shift, tint_shift, saturation] all normalised to [-1, 1].
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import COLOR_PARAM_MODEL_NAME, COLOR_PARAM_NAMES


def build_param_model(pretrained: bool = True, backbone_name: str | None = None) -> nn.Module:
    import timm
    name = backbone_name or COLOR_PARAM_MODEL_NAME
    # ViT-based models: use CLS token ("avg" adds fc_norm that conflicts with
    # DINOv2/SigLIP pretrained keys), and dynamic_img_size so 512px input works.
    is_vit = "vit" in name or "swin" in name
    pool   = "token" if is_vit else "avg"
    extra  = {"dynamic_img_size": True} if is_vit else {}
    backbone = timm.create_model(name, pretrained=pretrained,
                                  num_classes=0, global_pool=pool, **extra)
    head = nn.Sequential(
        nn.Linear(backbone.num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(COLOR_PARAM_NAMES)),
        nn.Tanh(),
    )

    class ParamRegressor(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            return self.head(self.backbone(x))

    return ParamRegressor(backbone, head)

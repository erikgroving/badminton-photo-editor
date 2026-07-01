"""
Culling classifier: any timm backbone fine-tuned for keep(1) / reject(0).

The asymmetric loss (fn_weight) makes false negatives — missing a "keep" photo —
far more costly than false positives.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CULL_MODEL_NAME


def build_model(backbone: str = CULL_MODEL_NAME, pretrained: bool = True,
                dynamic_img_size: bool = False) -> nn.Module:
    import timm
    extra = {"dynamic_img_size": True} if dynamic_img_size else {}
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=1, **extra)
    return model


class AsymmetricBCELoss(nn.Module):
    """
    BCE loss where predicting "reject" on a truly-kept photo is penalized
    fn_weight times more than the reverse mistake.
    """

    def __init__(self, fn_weight: float):
        super().__init__()
        self.fn_weight = fn_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = torch.tensor([self.fn_weight], device=logits.device, dtype=logits.dtype)
        return nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(1), targets.float(), pos_weight=pw
        )

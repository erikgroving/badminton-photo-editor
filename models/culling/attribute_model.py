"""
Attribute-aware culling model.

Architecture:
  CNN backbone  (pretrained, fine-tuned)  →  feat_dim-d vector
  Attribute MLP (8 scalars)               →  32-d vector
  Fusion MLP    (feat_dim + 32)           →  64 → 1  (logit)

The CNN learns "is this a decisive, well-composed action shot?"
The attribute branch gives explicit channels for blur, exposure,
face quality, etc. so those reasons don't have to be re-discovered
from pixels.

Interpretability: after training, the fusion MLP weights show which
attributes matter most — readable as a table in the review UI.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.attribute_extractor import ATTR_DIM


class AttributeAwareCullingModel(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        import timm

        # CNN backbone — remove classification head, keep pooled features
        self.cnn = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                     global_pool="avg")
        feat_dim = self.cnn.num_features

        # Small MLP to project attribute scalars into the same embedding space
        self.attr_mlp = nn.Sequential(
            nn.Linear(ATTR_DIM, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim + 32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, imgs: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        """
        imgs:  (B, 3, H, W)
        attrs: (B, ATTR_DIM)  — normalised attribute scalars
        Returns (B, 1) logits.
        """
        cnn_feat  = self.cnn(imgs)                  # (B, feat_dim)
        attr_feat = self.attr_mlp(attrs)            # (B, 32)
        fused     = torch.cat([cnn_feat, attr_feat], dim=1)
        return self.fusion(fused)                   # (B, 1)

    def attribute_importance(self, device: torch.device | None = None) -> dict[str, float]:
        """
        Returns a dict of {attr_name: importance_score} derived from the L2
        norm of the first attr_mlp layer weights.  Useful for logging / UI.
        """
        from data.attribute_extractor import ATTR_NAMES
        w = self.attr_mlp[0].weight   # (32, ATTR_DIM)
        scores = w.norm(dim=0)        # (ATTR_DIM,)
        scores = scores / scores.sum()
        return {name: float(s) for name, s in zip(ATTR_NAMES, scores.tolist())}

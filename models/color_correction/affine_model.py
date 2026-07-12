"""
Global 3x4 color-affine model — the production color-correction approach.

Predicts a per-photo 3x4 matrix M from the full downsampled frame; the
correction is rgb' = M[:, :3] @ rgb + M[:, 3] applied in LINEAR RGB.
Trained in image space (Lab L1 vs the photographer's edited JPGs) by
train_color_affine.py. Measured val dE2000 = 5.53 vs 13.08 do-nothing
(slider-renderer ceiling was 12.46).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

AFFINE_INPUT_SIZE = 256   # full-frame model input


class AffinePredictor(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0"):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=False,
                                          num_classes=0, global_pool="avg")
        self.head = nn.Linear(self.backbone.num_features, 12)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():   # bias -> identity affine
            self.head.bias.copy_(torch.tensor(
                [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]))

    def forward(self, x):
        return self.head(self.backbone(x)).view(-1, 3, 4)


def srgb_to_lab_t(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: (...,3) in [0,1] -> Lab. Differentiable."""
    c = rgb.clamp(0.0, 1.0)
    lin = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]],
                     dtype=rgb.dtype, device=rgb.device)
    xyz = lin @ M.T
    wp = torch.tensor([0.95047, 1.0, 1.08883], dtype=rgb.dtype, device=rgb.device)
    t = (xyz / wp).clamp(min=1e-6)
    eps = 216 / 24389
    f = torch.where(t > eps, t ** (1 / 3), (24389 / 27 * t + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)


def apply_affine_t(img: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """img: [B,3,H,W] in [0,1] sRGB; M: [B,3,4]. Affine in LINEAR RGB."""
    x = img.clamp(1e-6, 1.0).permute(0, 2, 3, 1) ** 2.2
    y = torch.einsum("bij,bhwj->bhwi", M[:, :, :3], x) + M[:, None, None, :, 3]
    return (y.clamp(1e-6, 4.0) ** (1 / 2.2)).permute(0, 3, 1, 2).clamp(0.0, 1.0)


def apply_affine_pil(img: Image.Image, M: np.ndarray) -> Image.Image:
    """Apply a 3x4 affine (numpy) to a PIL image at native resolution."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    lin = np.clip(arr, 1e-6, 1.0) ** 2.2
    out = lin @ M[:, :3].T + M[:, 3]
    out = np.clip(out, 1e-6, 4.0) ** (1 / 2.2)
    return Image.fromarray((np.clip(out, 0.0, 1.0) * 255).astype(np.uint8))


def predict_affine(model: AffinePredictor, full_frame: Image.Image,
                   device) -> np.ndarray:
    """Full developed frame (any size) -> 3x4 matrix (numpy)."""
    inp = full_frame.convert("RGB").resize((AFFINE_INPUT_SIZE, AFFINE_INPUT_SIZE),
                                           Image.LANCZOS)
    t = torch.from_numpy(np.asarray(inp, dtype=np.float32) / 255.0
                         ).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        M = model(t)[0].cpu().numpy()
    return M


def load_affine_model(ckpt_path: str | Path, device) -> AffinePredictor | None:
    p = Path(ckpt_path)
    if not p.exists():
        return None
    ck = torch.load(p, map_location=device, weights_only=False)
    model = AffinePredictor(ck.get("backbone", "efficientnet_b0"))
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    return model

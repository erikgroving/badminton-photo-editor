"""
Crop regressor: timm backbone + box head + optional angle regression head.

Box head:   [x1, y1, x2, y2] normalized to [0, 1], sigmoid activated.
Angle head: continuous rotation angle in degrees / 90  (raw linear output,
            no activation).  Multiply by 90 at inference to get degrees.

GT angles cluster at 0° (landscape) and 90° (portrait), with real ±5–13°
tilt corrections making up ~2.5% of images.  Binary classification is
insufficient — we regress the actual angle.

GT format (from data/crop_gt.json):
    box = [x1, y1, x2, y2] normalized to [0, 1]  (in raw thumbnail space)
    angle_deg: rotation detected by SIFT homography
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CROP_MODEL_NAME

ANGLE_SCALE = 90.0  # normalize: angle_norm = angle_deg / ANGLE_SCALE

# ── Conditioning feature builders (used by CombinedDINOv2 / exp6) ──────────────

def build_rich_features(union_bbox: torch.Tensor,
                        primary_bbox: torch.Tensor) -> torch.Tensor:
    """[B,4] union + [B,4] primary → [B, 13] analytical player features."""
    eps = 1e-6
    ux1, uy1, ux2, uy2 = union_bbox.unbind(1)
    px1, py1, px2, py2 = primary_bbox.unbind(1)
    uw = (ux2 - ux1).clamp(min=0.0)
    uh = (uy2 - uy1).clamp(min=0.0)
    cx = ux1 + uw / 2.0
    cy = uy1 + uh / 2.0
    u_area = uw * uh
    pw = (px2 - px1).clamp(min=0.0)
    ph = (py2 - py1).clamp(min=0.0)
    p_area = pw * ph
    dx = cx - 0.5
    dy = cy - 0.5
    dist = (dx ** 2 + dy ** 2).sqrt()
    vert_third = (cy / (1.0 / 3.0 + eps)).clamp(max=2.0) / 2.0
    area_ratio = u_area / (p_area + eps)
    has_two = (area_ratio > 1.4).float()
    return torch.stack([
        cx, cy, uw, uh, u_area,
        uw / (uh + eps),
        dx, dy, dist,
        vert_third, u_area,
        (area_ratio - 1.0).clamp(min=0.0),
        has_two,
    ], dim=1)  # [B, 13]


def build_rot_features(player_bbox: torch.Tensor) -> torch.Tensor:
    """[B,4] primary bbox → [B, 7] rule-of-thirds distance features."""
    x1, y1, x2, y2 = player_bbox.unbind(1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    r1, r2 = 1 / 3, 2 / 3
    dx = torch.minimum(torch.abs(cx - r1), torch.abs(cx - r2))
    dy = torch.minimum(torch.abs(cy - r1), torch.abs(cy - r2))
    area = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)).sqrt()
    return torch.stack([dx, dy, cx - r1, cx - r2, cy - r1, cy - r2, area], dim=1)  # [B, 7]


def compute_region_stats(img: Image.Image, grid: int = 3) -> list:
    """PIL image → 27 floats: 3×3 grid × (brightness, edge_density, saturation)."""
    h, w = img.height, img.width
    img_np = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    gray = img_np.mean(axis=2)
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    edges = gx + gy
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    sat = np.where(maxc > 0, (maxc - minc) / (maxc + 1e-6), 0.0)
    feats = []
    rh, rw = h // grid, w // grid
    for i in range(grid):
        for j in range(grid):
            feats.extend([
                float(gray [i*rh:(i+1)*rh, j*rw:(j+1)*rw].mean()),
                float(np.clip(edges[i*rh:(i+1)*rh, j*rw:(j+1)*rw].mean(), 0.0, 1.0)),
                float(sat  [i*rh:(i+1)*rh, j*rw:(j+1)*rw].mean()),
            ])
    return feats  # 27 values


def build_crop_model(backbone: str = CROP_MODEL_NAME, pretrained: bool = True,
                     use_angle_head: bool = False,
                     use_player_bbox: bool = False) -> nn.Module:
    import timm
    try:
        backbone_model = timm.create_model(backbone, pretrained=pretrained,
                                           num_classes=0, global_pool="avg")
    except RuntimeError:
        backbone_model = timm.create_model(backbone, pretrained=False,
                                           num_classes=0, global_pool="avg")
        if pretrained:
            _load_backbone_from_culling_ckpt(backbone_model, backbone)
    in_features      = backbone_model.num_features
    _use_angle       = use_angle_head
    _use_player_bbox = use_player_bbox
    _player_emb_dim  = 32 if _use_player_bbox else 0
    _head_in         = in_features + _player_emb_dim

    class CropRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone        = backbone_model
            self.use_angle_head  = _use_angle
            self.use_player_bbox = _use_player_bbox
            if _use_player_bbox:
                # Normalized player union bbox [x1,y1,x2,y2] → embedding, so the model
                # learns Jay's portrait/landscape composition from player position rather
                # than relying on hardcoded aspect ratios.
                self.player_encoder = nn.Sequential(
                    nn.Linear(4, _player_emb_dim),
                    nn.ReLU(),
                )
            self.box_head = nn.Sequential(
                nn.Linear(_head_in, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 4),
                nn.Sigmoid(),
            )
            if _use_angle:
                self.angle_head = nn.Sequential(
                    nn.Linear(_head_in, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                )

        def forward(self, x: torch.Tensor,
                    player_bbox: torch.Tensor | None = None):
            feats = self.backbone(x)
            if self.use_player_bbox and player_bbox is not None:
                feats = torch.cat([feats, self.player_encoder(player_bbox)], dim=1)
            box = self.box_head(feats)
            if self.use_angle_head:
                return box, self.angle_head(feats).squeeze(-1)
            return box

        def set_grad_checkpointing(self, enable: bool = True) -> None:
            if hasattr(self.backbone, "set_grad_checkpointing"):
                self.backbone.set_grad_checkpointing(enable=enable)

    return CropRegressor()


class CropLoss(nn.Module):
    """
    Combined loss:
        box_loss      = alpha * SmoothL1  +  (1-alpha) * (1-IoU)
        angle_loss    = SmoothL1(angle_pred, angle_target)   # targets normalized by /90
        player_loss   = hinge penalty when predicted crop clips any edge of the player box
        total         = (1-angle_weight) * box_loss
                        + angle_weight * angle_loss          # only when angle head active
                        + player_weight * player_loss        # only when player_bbox supplied

    Player-coverage penalty:
        Applied on raw sigmoid output (has gradient). For each sample with a detected
        player, penalises any edge of the predicted crop that clips the player box:

            clip_left   = max(0, pred_x1 - player_x1)   # crop left cuts into player
            clip_top    = max(0, pred_y1 - player_y1)   # crop top clips player head
            clip_right  = max(0, player_x2 - pred_x2)   # crop right clips player
            clip_bottom = max(0, player_y2 - pred_y2)   # crop bottom clips player feet

        penalty = mean(clip_left + clip_top + clip_right + clip_bottom)
                  over samples where a player was detected (bbox != [0,0,0,0]).

        Samples with no detection (all-zero bbox) are explicitly masked out.
    """
    def __init__(self, alpha: float = 0.5, angle_weight: float = 0.25,
                 player_weight: float = 0.5, player_margin: float = 0.0,
                 ar_weight: float = 0.0, use_ciou: bool = False):
        super().__init__()
        self.alpha          = alpha
        self.angle_weight   = angle_weight
        self.player_weight  = player_weight
        self.player_margin  = player_margin
        self.ar_weight      = ar_weight
        self.use_ciou       = use_ciou

    def forward(self, box_pred: torch.Tensor, box_target: torch.Tensor,
                angle_pred: torch.Tensor | None = None,
                angle_target: torch.Tensor | None = None,
                player_bbox: torch.Tensor | None = None) -> torch.Tensor:
        sl1      = nn.functional.smooth_l1_loss(box_pred, box_target)
        iou      = (_box_ciou_loss(box_pred, box_target) if self.use_ciou
                    else _box_iou_loss(box_pred, box_target))
        box_loss = self.alpha * sl1 + (1.0 - self.alpha) * iou

        if angle_pred is not None and angle_target is not None:
            angle_loss = nn.functional.smooth_l1_loss(
                angle_pred, angle_target.to(angle_pred.device)
            )
            total = (1.0 - self.angle_weight) * box_loss + self.angle_weight * angle_loss
        else:
            total = box_loss

        if player_bbox is not None and self.player_weight > 0.0:
            # Only penalise samples where a player was actually detected.
            has_player = player_bbox.sum(dim=1) > 0.0          # [B] bool
            if has_player.any():
                pb = player_bbox[has_player]                    # [N, 4]
                pp = box_pred[has_player]                       # [N, 4]
                px1, py1, px2, py2 = pb.unbind(1)
                cx1, cy1, cx2, cy2 = pp.unbind(1)
                clip = (
                    (cx1 - px1).clamp(min=0.0) +               # left edge clips player
                    (cy1 - py1 - self.player_margin).clamp(min=0.0) +  # top clips head
                    (px2 - cx2).clamp(min=0.0) +               # right edge clips player
                    (py2 + self.player_margin - cy2).clamp(min=0.0)    # bottom clips feet
                )
                total = total + self.player_weight * clip.mean()

        if self.ar_weight > 0.0:
            w_pred = (box_pred[:, 2] - box_pred[:, 0]).clamp(min=1e-6)
            h_pred = (box_pred[:, 3] - box_pred[:, 1]).clamp(min=1e-6)
            w_gt   = (box_target[:, 2] - box_target[:, 0]).clamp(min=1e-6)
            h_gt   = (box_target[:, 3] - box_target[:, 1]).clamp(min=1e-6)
            ar_loss = nn.functional.smooth_l1_loss(w_pred / h_pred, w_gt / h_gt)
            total = total + self.ar_weight * ar_loss

        return total


def _box_iou_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    ix1  = torch.max(pred[:, 0], gt[:, 0])
    iy1  = torch.max(pred[:, 1], gt[:, 1])
    ix2  = torch.min(pred[:, 2], gt[:, 2])
    iy2  = torch.min(pred[:, 3], gt[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pw   = (pred[:, 2] - pred[:, 0]).clamp(0)
    ph   = (pred[:, 3] - pred[:, 1]).clamp(0)
    gw   = (gt[:, 2]   - gt[:, 0]).clamp(0)
    gh   = (gt[:, 3]   - gt[:, 1]).clamp(0)
    union = pw * ph + gw * gh - inter
    return (1.0 - inter / (union + 1e-6)).mean()


def _box_ciou_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Complete IoU loss (Zheng et al. 2020): IoU − center-distance/enclosing-diag
    − aspect-ratio consistency.  Subsumes the ad-hoc ar_weight term."""
    eps = 1e-6
    ix1  = torch.max(pred[:, 0], gt[:, 0])
    iy1  = torch.max(pred[:, 1], gt[:, 1])
    ix2  = torch.min(pred[:, 2], gt[:, 2])
    iy2  = torch.min(pred[:, 3], gt[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pw   = (pred[:, 2] - pred[:, 0]).clamp(min=eps)
    ph   = (pred[:, 3] - pred[:, 1]).clamp(min=eps)
    gw   = (gt[:, 2]   - gt[:, 0]).clamp(min=eps)
    gh   = (gt[:, 3]   - gt[:, 1]).clamp(min=eps)
    union = pw * ph + gw * gh - inter
    iou   = inter / (union + eps)

    # center distance over enclosing-box diagonal
    pcx = (pred[:, 0] + pred[:, 2]) * 0.5
    pcy = (pred[:, 1] + pred[:, 3]) * 0.5
    gcx = (gt[:, 0]   + gt[:, 2])   * 0.5
    gcy = (gt[:, 1]   + gt[:, 3])   * 0.5
    rho2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2
    ex1  = torch.min(pred[:, 0], gt[:, 0])
    ey1  = torch.min(pred[:, 1], gt[:, 1])
    ex2  = torch.max(pred[:, 2], gt[:, 2])
    ey2  = torch.max(pred[:, 3], gt[:, 3])
    c2   = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + eps

    # aspect-ratio consistency
    import math
    v = (4.0 / math.pi ** 2) * (torch.atan(gw / gh) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha * v
    return (1.0 - ciou).mean()


def box_iou_numpy(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    ix1  = np.maximum(pred[:, 0], gt[:, 0])
    iy1  = np.maximum(pred[:, 1], gt[:, 1])
    ix2  = np.minimum(pred[:, 2], gt[:, 2])
    iy2  = np.minimum(pred[:, 3], gt[:, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    pw   = np.maximum(0, pred[:, 2] - pred[:, 0])
    ph   = np.maximum(0, pred[:, 3] - pred[:, 1])
    gw   = np.maximum(0, gt[:, 2]   - gt[:, 0])
    gh   = np.maximum(0, gt[:, 3]   - gt[:, 1])
    union = pw * ph + gw * gh - inter
    return inter / np.maximum(union, 1e-6)


class CombinedDINOv2(nn.Module):
    """
    DINOv2 ViT-B + 51-dim combined conditioning → box + angle.

    Conditioning: union_bbox[4] + rich_features[13] + rot_features[7] + img_stats[27] = 51-dim.
    Forward: model(x, union_bbox, primary_bbox, img_stats) → (box [B,4], angle_norm [B])
    """
    def __init__(self, backbone: nn.Module, backbone_dim: int,
                 cond_dim: int = 51, cond_emb_dim: int = 128):
        super().__init__()
        self.backbone     = backbone
        self.cond_dim     = cond_dim
        self.cond_emb_dim = cond_emb_dim
        self.use_player_bbox = True  # pipeline compatibility flag

        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.ReLU(),
            nn.Linear(128, cond_emb_dim),
            nn.ReLU(),
        )
        head_in = backbone_dim + cond_emb_dim
        self.box_head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )
        self.angle_head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.use_angle_head = True

    def forward(self, x: torch.Tensor,
                union_bbox: torch.Tensor,
                primary_bbox: torch.Tensor,
                img_stats: torch.Tensor):
        feats    = self.backbone(x)
        rich     = build_rich_features(union_bbox, primary_bbox)
        rot      = build_rot_features(primary_bbox)
        cond     = torch.cat([union_bbox, rich, rot, img_stats], dim=1)
        cond_emb = self.cond_encoder(cond)
        combined = torch.cat([feats, cond_emb], dim=1)
        box      = self.box_head(combined)
        angle    = self.angle_head(combined).squeeze(-1)
        return box, angle

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=enable)


def build_combined_crop_model(backbone: str = "vit_base_patch14_reg4_dinov2",
                               pretrained: bool = False,
                               cond_dim: int = 51,
                               cond_emb_dim: int = 128) -> CombinedDINOv2:
    import timm
    bb = timm.create_model(backbone, pretrained=pretrained,
                           num_classes=0, global_pool="avg")
    return CombinedDINOv2(bb, bb.num_features, cond_dim=cond_dim, cond_emb_dim=cond_emb_dim)


def _load_backbone_from_culling_ckpt(backbone_model: nn.Module, backbone_name: str) -> bool:
    """
    Initialize backbone weights from a culling checkpoint when timm pretrained
    loading fails due to a state-dict key mismatch (e.g. after a timm update).

    The culling model is a bare timm model (num_classes=1), so state dict keys are
    at the top level.  We drop the classifier head keys and remap any renamed
    layer-norm keys (timm renamed 'norm' → 'fc_norm' in a recent release).
    Returns True if successful.
    """
    from config import CHECKPOINTS_DIR
    ckpt_path = CHECKPOINTS_DIR / f"culling_{backbone_name.replace('/', '_')}.pt"
    if not ckpt_path.exists():
        print(f"  Warning: no culling checkpoint for {backbone_name}; using random init.")
        return False

    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    src = ck["model_state"]

    new_model_keys = set(backbone_model.state_dict().keys())
    uses_fc_norm   = any("fc_norm" in k for k in new_model_keys)
    backbone_state = {}
    for k, v in src.items():
        if k in ("head.weight", "head.bias"):
            continue
        mapped = k
        if uses_fc_norm and k.startswith("norm."):
            mapped = "fc_norm." + k[len("norm."):]
        backbone_state[mapped] = v

    missing, unexpected = backbone_model.load_state_dict(backbone_state, strict=False)
    matched = len(backbone_state) - len(unexpected)
    print(f"  Backbone init from culling ckpt: {matched}/{len(backbone_state)} keys matched  "
          f"missing={len(missing)}  unexpected={len(unexpected)}")
    return matched > 0


# ── Exp 7: pose keypoints + dual portrait/landscape heads ─────────────────────

# COCO 17-keypoint left/right swap pairs for horizontal flip augmentation
COCO_FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
N_KEYPOINTS = 17
POSE_DIM    = N_KEYPOINTS * 2   # x, y only; YOLO-Pose confidence is always 0.5 so we drop it


def flip_pose_kpts(kpts: list[float]) -> list[float]:
    """Mirror x coords and swap left/right pairs for hflip augmentation."""
    arr = list(kpts)
    for i in range(N_KEYPOINTS):
        arr[i * 2] = 1.0 - arr[i * 2]
    for a, b in COCO_FLIP_PAIRS:
        arr[a*2],   arr[b*2]   = arr[b*2],   arr[a*2]
        arr[a*2+1], arr[b*2+1] = arr[b*2+1], arr[a*2+1]
    return arr


class CombinedDINOv2Exp7(nn.Module):
    """
    DINOv2 + pose keypoints conditioning + separate portrait/landscape box heads.

    Conditioning (85-dim default):
        union_bbox[4] + rich_features[13] + rot_features[7] + img_stats[27] + pose_kpts[34]

    Two box heads (landscape / portrait) blended by a soft sigmoid gate derived
    from the predicted angle_norm.  angle_norm > 0.5 (>45°) routes to portrait.

    Compatible with dynamic_img_size=True backbones for 770px input.
    """
    def __init__(self, backbone: nn.Module, backbone_dim: int,
                 cond_dim: int = 51 + POSE_DIM,
                 cond_emb_dim: int = 128):
        super().__init__()
        self.backbone        = backbone
        self.cond_dim        = cond_dim
        self.cond_emb_dim    = cond_emb_dim
        self.use_player_bbox = True
        self.use_angle_head  = True

        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, cond_emb_dim),
            nn.ReLU(),
        )
        head_in = backbone_dim + cond_emb_dim

        def _head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(head_in, 256), nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 4), nn.Sigmoid(),
            )

        self.box_head_landscape = _head()
        self.box_head_portrait  = _head()
        self.angle_head = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self,
                x: torch.Tensor,
                union_bbox: torch.Tensor,
                primary_bbox: torch.Tensor,
                img_stats: torch.Tensor,
                pose_kpts: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        rich  = build_rich_features(union_bbox, primary_bbox)
        rot   = build_rot_features(primary_bbox)

        parts = [union_bbox, rich, rot, img_stats]
        if pose_kpts is not None:
            parts.append(pose_kpts)
        cond     = torch.cat(parts, dim=1)
        cond_emb = self.cond_encoder(cond)
        combined = torch.cat([feats, cond_emb], dim=1)

        angle_norm = self.angle_head(combined).squeeze(-1)      # [B]
        box_l      = self.box_head_landscape(combined)           # [B, 4]
        box_p      = self.box_head_portrait(combined)            # [B, 4]

        # Soft blend: portrait_w → 1 when angle_norm > 0.5  (portrait threshold = 45°)
        p_w = torch.sigmoid((angle_norm - 0.5) * 10).unsqueeze(-1)  # [B, 1]
        box = (1.0 - p_w) * box_l + p_w * box_p                     # [B, 4]

        return box, angle_norm

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=enable)


def _load_vit_pretrained_remapped(model: nn.Module, backbone: str,
                                  dynamic_img_size: bool = True) -> None:
    """Download pretrained ViT weights and remap norm→fc_norm for global_pool='avg'.

    When global_pool='avg', timm stores the final layer norm as 'fc_norm' instead
    of 'norm', but DINOv2 checkpoints on HuggingFace use 'norm'. We create a
    temporary model with global_pool='' to load the original keys, then remap.
    """
    import timm
    extra = {"dynamic_img_size": True} if dynamic_img_size else {}
    try:
        tmp = timm.create_model(backbone, pretrained=True,
                                num_classes=0, global_pool="", **extra)
        sd_src = tmp.state_dict()
        del tmp
    except Exception as e:
        print(f"  Pretrained load warning (using random init): {e}")
        return

    # Remap norm→fc_norm so it matches the global_pool='avg' model
    remapped = {}
    for k, v in sd_src.items():
        new_k = "fc_norm." + k[len("norm."):] if k.startswith("norm.") else k
        remapped[new_k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"  Pretrained {backbone}: loaded "
          f"(missing={len(missing)} unexpected={len(unexpected)})")


def build_exp7_model(backbone: str = "vit_large_patch14_reg4_dinov2",
                     pretrained: bool = False,
                     pose_dim: int = POSE_DIM,
                     cond_emb_dim: int = 128,
                     dynamic_img_size: bool = True) -> "CombinedDINOv2Exp7":
    import timm
    extra = {"dynamic_img_size": True} if dynamic_img_size else {}
    bb    = timm.create_model(backbone, pretrained=False,
                              num_classes=0, global_pool="avg", **extra)
    if pretrained:
        _load_vit_pretrained_remapped(bb, backbone, dynamic_img_size)
    return CombinedDINOv2Exp7(bb, bb.num_features,
                               cond_dim=51 + pose_dim,
                               cond_emb_dim=cond_emb_dim)


# ── Exp 8: spatial patch features + hard portrait/landscape head routing ───────

class CombinedDINOv2Exp8(nn.Module):
    """
    Exp7 + two improvements:

    1. Spatial features: concatenates CLS token with mean-pooled patch tokens,
       then projects to backbone_dim.  CLS captures global semantics; patch mean
       retains spatial layout information lost by global-avg-pool alone.

    2. Hard head routing during training: routes each sample to the correct
       portrait or landscape box head using the GT angle label (is_portrait flag).
       At inference is_portrait=None → soft sigmoid blend (same as exp7).
    """
    def __init__(self, backbone: nn.Module, backbone_dim: int,
                 cond_dim: int = 51 + POSE_DIM,
                 cond_emb_dim: int = 128):
        super().__init__()
        self.backbone        = backbone
        self.cond_dim        = cond_dim
        self.cond_emb_dim    = cond_emb_dim
        self.use_player_bbox = True
        self.use_angle_head  = True

        # Project [cls; patch_mean] → backbone_dim
        self.spatial_proj = nn.Sequential(
            nn.Linear(2 * backbone_dim, backbone_dim),
            nn.LayerNorm(backbone_dim),
            nn.GELU(),
        )
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, 256), nn.ReLU(),
            nn.Linear(256, cond_emb_dim), nn.ReLU(),
        )
        head_in = backbone_dim + cond_emb_dim

        def _head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(head_in, 256), nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 4), nn.Sigmoid(),
            )

        self.box_head_landscape = _head()
        self.box_head_portrait  = _head()
        self.angle_head = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self,
                x: torch.Tensor,
                union_bbox: torch.Tensor,
                primary_bbox: torch.Tensor,
                img_stats: torch.Tensor,
                pose_kpts: torch.Tensor | None = None,
                is_portrait: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # --- spatial feature extraction ---
        seq        = self.backbone.forward_features(x)    # [B, N_total, D]
        n_prefix   = getattr(self.backbone, "num_prefix_tokens", 1)  # CLS + reg tokens
        cls_feat   = seq[:, 0]                            # [B, D] CLS token
        patch_feat = seq[:, n_prefix:].mean(dim=1)        # [B, D] avg of patch tokens
        feats      = self.spatial_proj(
            torch.cat([cls_feat, patch_feat], dim=1))     # [B, D]

        # --- conditioning ---
        rich  = build_rich_features(union_bbox, primary_bbox)
        rot   = build_rot_features(primary_bbox)
        parts = [union_bbox, rich, rot, img_stats]
        if pose_kpts is not None:
            parts.append(pose_kpts)
        cond_emb = self.cond_encoder(torch.cat(parts, dim=1))
        combined = torch.cat([feats, cond_emb], dim=1)

        angle_norm = self.angle_head(combined).squeeze(-1)   # [B]
        box_l      = self.box_head_landscape(combined)        # [B, 4]
        box_p      = self.box_head_portrait(combined)         # [B, 4]

        if is_portrait is not None:
            # Hard routing: GT label selects the head (training only)
            mask = is_portrait.unsqueeze(-1)                 # [B, 1] bool
            box  = torch.where(mask, box_p, box_l)
        else:
            # Soft blend at inference
            p_w = torch.sigmoid((angle_norm - 0.5) * 10).unsqueeze(-1)
            box = (1.0 - p_w) * box_l + p_w * box_p

        return box, angle_norm

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=enable)


def build_exp8_model(backbone: str = "vit_large_patch14_reg4_dinov2",
                     pretrained: bool = False,
                     pose_dim: int = POSE_DIM,
                     cond_emb_dim: int = 128,
                     dynamic_img_size: bool = True) -> "CombinedDINOv2Exp8":
    import timm
    extra = {"dynamic_img_size": True} if dynamic_img_size else {}
    bb    = timm.create_model(backbone, pretrained=False,
                              num_classes=0, global_pool="avg", **extra)
    if pretrained:
        _load_vit_pretrained_remapped(bb, backbone, dynamic_img_size)
    return CombinedDINOv2Exp8(bb, bb.num_features,
                               cond_dim=51 + pose_dim,
                               cond_emb_dim=cond_emb_dim)


# ── Exp 9a: multi-layer feature fusion + attentive pooling ─────────────────────

class AttentivePooler(nn.Module):
    """Learned-query cross-attention pooling over patch tokens.

    Mean-pooling destroys spatial selectivity; a small set of learned queries
    lets each query attend to the regions relevant to its output (e.g. one per
    box edge).  ~1M params for D=1024, n_queries=4.
    """
    def __init__(self, dim: int, n_queries: int = 4, n_heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, dim) * 0.02)
        self.attn    = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm    = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:   # [B, N, D]
        q      = self.queries.expand(tokens.shape[0], -1, -1)  # [B, Q, D]
        out, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.norm(out)                                  # [B, Q, D]


class CombinedDINOv2Exp9(nn.Module):
    """
    Exp8 + two evidence-backed feature upgrades (DINOv2 paper Table 11 shows
    ~13% dense-regression gain from multi-layer features; attentive-probing
    literature shows learned-query pooling beats mean pooling):

    1. Multi-layer fusion: patch tokens from 4 intermediate blocks are
       concatenated channel-wise and projected back to backbone_dim.
    2. Attentive pooling: 4 learned queries cross-attend the fused tokens
       instead of mean-pooling them.

    Head structure (cond encoder, dual portrait/landscape box heads with hard
    GT routing at training / soft blend at inference, angle head) is identical
    to exp8 so training and evaluation code is shared.
    """
    LAYER_INDICES = (5, 11, 17, 23)   # for ViT-L (24 blocks)

    def __init__(self, backbone: nn.Module, backbone_dim: int,
                 cond_dim: int = 51 + POSE_DIM,
                 cond_emb_dim: int = 128,
                 n_queries: int = 4):
        super().__init__()
        self.backbone        = backbone
        self.cond_dim        = cond_dim
        self.cond_emb_dim    = cond_emb_dim
        self.use_player_bbox = True
        self.use_angle_head  = True

        n_layers = len(self.LAYER_INDICES)
        self.fuse_proj = nn.Sequential(
            nn.Linear(n_layers * backbone_dim, backbone_dim),
            nn.LayerNorm(backbone_dim),
            nn.GELU(),
        )
        self.pool = AttentivePooler(backbone_dim, n_queries=n_queries)
        self.pool_proj = nn.Sequential(
            nn.Linear(n_queries * backbone_dim, backbone_dim),
            nn.LayerNorm(backbone_dim),
            nn.GELU(),
        )
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, 256), nn.ReLU(),
            nn.Linear(256, cond_emb_dim), nn.ReLU(),
        )
        head_in = backbone_dim + cond_emb_dim

        def _head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(head_in, 256), nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 4), nn.Sigmoid(),
            )

        self.box_head_landscape = _head()
        self.box_head_portrait  = _head()
        self.angle_head = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self,
                x: torch.Tensor,
                union_bbox: torch.Tensor,
                primary_bbox: torch.Tensor,
                img_stats: torch.Tensor,
                pose_kpts: torch.Tensor | None = None,
                is_portrait: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # --- multi-layer spatial features ---
        inters = self.backbone.forward_intermediates(
            x, indices=list(self.LAYER_INDICES), norm=True,
            output_fmt="NLC", intermediates_only=True)         # 4 × [B, N, D]
        tokens = self.fuse_proj(torch.cat(inters, dim=-1))     # [B, N, D]
        pooled = self.pool(tokens)                             # [B, Q, D]
        feats  = self.pool_proj(pooled.flatten(1))             # [B, D]

        # --- conditioning (same as exp8) ---
        rich  = build_rich_features(union_bbox, primary_bbox)
        rot   = build_rot_features(primary_bbox)
        parts = [union_bbox, rich, rot, img_stats]
        if pose_kpts is not None:
            parts.append(pose_kpts)
        cond_emb = self.cond_encoder(torch.cat(parts, dim=1))
        combined = torch.cat([feats, cond_emb], dim=1)

        angle_norm = self.angle_head(combined).squeeze(-1)
        box_l      = self.box_head_landscape(combined)
        box_p      = self.box_head_portrait(combined)

        if is_portrait is not None:
            box = torch.where(is_portrait.unsqueeze(-1), box_p, box_l)
        else:
            p_w = torch.sigmoid((angle_norm - 0.5) * 10).unsqueeze(-1)
            box = (1.0 - p_w) * box_l + p_w * box_p

        return box, angle_norm

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=enable)


def build_exp9_model(backbone: str = "vit_large_patch14_reg4_dinov2",
                     pretrained: bool = False,
                     pose_dim: int = POSE_DIM,
                     cond_emb_dim: int = 128,
                     dynamic_img_size: bool = True) -> "CombinedDINOv2Exp9":
    import timm
    extra = {"dynamic_img_size": True} if dynamic_img_size else {}
    bb    = timm.create_model(backbone, pretrained=False,
                              num_classes=0, global_pool="avg", **extra)
    if pretrained:
        _load_vit_pretrained_remapped(bb, backbone, dynamic_img_size)
    return CombinedDINOv2Exp9(bb, bb.num_features,
                               cond_dim=51 + pose_dim,
                               cond_emb_dim=cond_emb_dim)

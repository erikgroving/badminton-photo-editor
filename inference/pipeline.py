"""
End-to-end batch inference pipeline.

Given a folder of CR3 raw files, runs:
  1. Culling   — keep / reject each photo
  2. Cropping  — predict crop box + rotation for kept photos
  3. Color     — apply param model or U-Net correction
  4. Export    — save final JPEGs to output_dir

Usage:
    python -m inference.pipeline --input /path/to/raws --output /path/to/out
    python -m inference.pipeline --input /path/to/raws --output /path/to/out --color-model unet
    python -m inference.pipeline --input /path/to/raws --output /path/to/out --dry-run
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHECKPOINTS_DIR, COLOR_PARAM_CKPT, COLOR_SIZE, COLOR_UNET_CKPT, CROP_CKPT,
    CULL_MODEL_NAME, DEVELOP_SIZE, JUDGE_CKPT, JUDGE_MODEL_NAME, THUMB_SIZE,
)
from data.raw_reader import develop_raw, extract_thumbnail, extract_thumbnail_ar, get_raw_flip
from inference.apply_params import apply_param_vec
from models.color_correction.param_model import build_param_model
from models.color_correction.unet import ColorUNet
from models.cropping.model import (
    build_crop_model, build_combined_crop_model, compute_region_stats, CombinedDINOv2,
)
from models.culling.model import build_model as build_cull_model
from models.judge.model import build_model as build_judge


# ── Model loaders ──────────────────────────────────────────────────────────────

def _load_cull(device):
    ckpt_files = list(CHECKPOINTS_DIR.glob("culling_*.pt"))
    if not ckpt_files:
        raise FileNotFoundError("No culling checkpoint found. Train first.")
    # Pick checkpoint with lowest asym_cost
    best = None
    for f in ckpt_files:
        d = torch.load(f, map_location="cpu")
        cost = d.get("metrics", {}).get("asym_cost", float("inf"))
        if best is None or cost < best[1]:
            best = (f, cost, d)
    ckpt             = best[2]
    backbone         = ckpt.get("backbone", CULL_MODEL_NAME)
    dynamic_img_size = ckpt.get("dynamic_img_size", False)
    model = build_cull_model(backbone=backbone, pretrained=False,
                             dynamic_img_size=dynamic_img_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    input_size = ckpt.get("input_size")
    if input_size is None:
        is_vit = backbone and ("vit" in backbone or "swin" in backbone)
        input_size = 518 if is_vit else THUMB_SIZE[0]
    model._input_size = input_size
    threshold = ckpt.get("threshold", 0.5)
    print(f"Culling model: {backbone}  input={input_size}  threshold={threshold:.3f}  (asym_cost={best[1]:.1f})")
    return model, threshold


def _load_crop(device):
    ckpt_files = list(CHECKPOINTS_DIR.glob("cropping_*.pt"))
    if not ckpt_files:
        raise FileNotFoundError("No crop checkpoint found. Train first.")

    # Prefer exp6 explicitly — intentionally trained best model.
    # Fallback: select by median_iou but only from test-split evals (n<300),
    # excluding full-dataset checkpoints with inflated metrics.
    exp6_path = CHECKPOINTS_DIR / "cropping_angle_vit_base_patch14_reg4_dinov2_exp6_combined.pt"
    if exp6_path.exists():
        best = exp6_path
    else:
        def _score(f):
            ck = torch.load(f, map_location="cpu", weights_only=False)
            m  = ck.get("metrics", {})
            if m.get("n", 9999) > 300:
                return -1.0
            return m.get("median_iou", 0.0)
        best = max(ckpt_files, key=_score)

    ckpt       = torch.load(best, map_location="cpu", weights_only=False)
    backbone   = ckpt.get("backbone", "resnet50")
    input_size = int(ckpt.get("input_size", 518))
    exp_tag    = ckpt.get("exp", "")

    if exp_tag == "exp6_combined":
        model = build_combined_crop_model(
            backbone=backbone, pretrained=False,
            cond_dim=ckpt.get("cond_dim", 51),
            cond_emb_dim=ckpt.get("cond_emb_dim", 128),
        ).to(device)
        print(f"Crop model: {backbone} [exp6 combined]  input={input_size}  "
              f"(median_iou={ckpt['metrics'].get('median_iou',0):.3f})")
    else:
        use_angle = ckpt.get("use_rot_head", ckpt.get("use_angle_head", False))
        use_pb    = ckpt.get("use_player_bbox", False)
        model = build_crop_model(backbone=backbone, pretrained=False,
                                 use_angle_head=use_angle,
                                 use_player_bbox=use_pb).to(device)
        pb_str = "  player-bbox" if use_pb else ""
        print(f"Crop model: {backbone}  input={input_size}  angle={use_angle}{pb_str}  "
              f"(median_iou={ckpt['metrics'].get('median_iou',0):.3f})")

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model._input_size = input_size
    return model


def _load_yolo():
    from ultralytics import YOLO
    bundled = CHECKPOINTS_DIR / "yolo11n.pt"
    return YOLO(str(bundled) if bundled.exists() else "yolo11n.pt")


def _detect_players(yolo_model, thumb: Image.Image) -> tuple[list, list]:
    """Return (union_bbox, primary_bbox) as [x1,y1,x2,y2] normalized [0,1]. Zeros if none."""
    w, h = thumb.size
    min_area = w * h * 0.01
    results = yolo_model(thumb, classes=[0], verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes.xyxy.cpu().tolist():
            x1, y1, x2, y2 = map(float, box)
            if (x2 - x1) * (y2 - y1) >= min_area:
                boxes.append([x1/w, y1/h, x2/w, y2/h])
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]
    primary = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
    if len(boxes) == 1:
        union = primary
    else:
        union = [
            min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes),
        ]
    return union, primary


def _load_color_param(device):
    ckpt_files = list(CHECKPOINTS_DIR.glob("color_param_*.pt"))
    # Fall back to legacy exact name
    if not ckpt_files and COLOR_PARAM_CKPT.exists():
        ckpt_files = [COLOR_PARAM_CKPT]
    if not ckpt_files:
        raise FileNotFoundError(f"No color param checkpoint in {CHECKPOINTS_DIR}. Train first.")
    # Prefer player-crop trained checkpoints (match inference input); fall back to val_l1
    def _ckpt_key(f):
        d = torch.load(f, map_location="cpu")
        player_crop = 0 if d.get("player_crop") else 1   # 0 sorts first
        val_l1 = d.get("metrics", {}).get("val_l1", float("inf"))
        return (player_crop, val_l1)
    best = min(ckpt_files, key=_ckpt_key)
    ckpt     = torch.load(best, map_location="cpu")
    backbone = ckpt.get("backbone")
    model    = build_param_model(pretrained=False, backbone_name=backbone).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # ViT/DINOv2 requires H and W divisible by patch size 14; use 518 = 14×37
    is_vit = backbone and ("vit" in backbone or "swin" in backbone)
    model._input_size = (518, 518) if is_vit else COLOR_SIZE
    print(f"Color model: {backbone}  input={model._input_size[0]}  (val_l1={ckpt['metrics'].get('val_l1', '?'):.4f})")
    return model


def _load_color_unet(device):
    if not COLOR_UNET_CKPT.exists():
        raise FileNotFoundError(f"No U-Net checkpoint at {COLOR_UNET_CKPT}. Train first.")
    ckpt  = torch.load(COLOR_UNET_CKPT, map_location="cpu")
    model = ColorUNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _load_judge(device):
    ckpt_files = list(CHECKPOINTS_DIR.glob("judge_*.pt"))
    if not ckpt_files:
        return None
    best = max(ckpt_files, key=lambda f: torch.load(f, map_location="cpu").get("metrics", {}).get("accuracy", 0))
    ckpt     = torch.load(best, map_location="cpu")
    backbone = ckpt.get("backbone", JUDGE_MODEL_NAME)
    model    = build_judge(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Judge model: {backbone}  (accuracy={ckpt['metrics'].get('accuracy',0):.4f})")
    return model


# ── Orientation coordinate helpers ─────────────────────────────────────────────
# develop_raw() uses rawpy.postprocess() which applies camera orientation, so portrait
# shots come out as portrait (e.g. 1027×1536).  extract_thumbnail_ar() does NOT apply
# orientation — it always returns the sensor-native landscape thumbnail (e.g. 512×341).
# The crop model was trained on unoriented thumbnails, so its predictions are in
# sensor-native (landscape) coordinates.  These helpers convert between spaces.

def _sensor_native_box(box4: list, flip: int) -> list:
    """
    Transform a bbox [x1,y1,x2,y2] normalised [0,1] from developed-image
    coordinate space into sensor-native (thumbnail) coordinate space.
    Inverse of _developed_image_box().
    """
    x1, y1, x2, y2 = box4
    if flip == 5:   # portrait: 90° CW to display → inverse is 90° CCW
        return [y1, 1-x2, y2, 1-x1]
    if flip == 6:   # portrait: 90° CCW to display → inverse is 90° CW
        return [1-y2, x1, 1-y1, x2]
    if flip == 3:   # 180°, self-inverse
        return [1-x2, 1-y2, 1-x1, 1-y1]
    return list(box4)  # flip==0: same space


def _developed_image_box(box4: list, flip: int) -> list:
    """
    Transform a bbox [x1,y1,x2,y2] normalised [0,1] from sensor-native
    (thumbnail/landscape) space into developed-image coordinate space.
    """
    x1, y1, x2, y2 = box4
    if flip == 5:   # 90° CW to get portrait
        return [1-y2, x1, 1-y1, x2]
    if flip == 6:   # 90° CCW to get portrait
        return [y1, 1-x2, y2, 1-x1]
    if flip == 3:   # 180°
        return [1-x2, 1-y2, 1-x1, 1-y1]
    return list(box4)


# ── Image transforms ───────────────────────────────────────────────────────────

_IMAGENET_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _pil_to_tensor(img: Image.Image, device) -> torch.Tensor:
    return _IMAGENET_TF(img).unsqueeze(0).to(device)


# ── Single-image inference ─────────────────────────────────────────────────────

@torch.no_grad()
def predict_cull(model, threshold, raw_path, device) -> tuple[bool, float]:
    """Returns (keep: bool, confidence: float)."""
    img = extract_thumbnail(raw_path, size=THUMB_SIZE)
    cull_sz = getattr(model, '_input_size', THUMB_SIZE[0])
    if cull_sz != img.size[0]:
        img = img.resize((cull_sz, cull_sz), Image.LANCZOS)
    prob = torch.sigmoid(model(_pil_to_tensor(img, device))).item()
    return prob >= threshold, prob


@torch.no_grad()
def predict_crop(model, raw_path, device, img_size=None,
                 player_bbox: list | None = None,
                 primary_bbox: list | None = None) -> tuple[list, float]:
    """Returns (crop_box: [x,y,w,h] in pixels, angle_deg: float).

    img_size:     actual (W, H) of the orientation-corrected developed image.
    player_bbox:  union bbox [x1,y1,x2,y2] normalised to *developed-image* space.
    primary_bbox: largest-player bbox, same space.  None → falls back to player_bbox.

    Internally: the model was trained on sensor-native (landscape) thumbnails, so
    player bboxes are converted to sensor-native space before the forward pass, and
    the predicted box is converted back to developed-image space before returning.
    """
    thumb      = extract_thumbnail_ar(raw_path, max_size=512)
    input_size = getattr(model, '_input_size', 518)
    resized    = thumb.resize((input_size, input_size), Image.LANCZOS)
    img_tensor = _pil_to_tensor(resized, device)
    flip       = get_raw_flip(raw_path)

    # Convert player bboxes from developed-image space → sensor-native space so
    # conditioning matches the training distribution (bboxes cached from thumbnails).
    def _to_sn(bbox: list | None) -> list:
        if bbox is None:
            return [0.0, 0.0, 0.0, 0.0]
        return _sensor_native_box(bbox, flip)

    if isinstance(model, CombinedDINOv2):
        union_t   = torch.tensor([_to_sn(player_bbox)], dtype=torch.float32).to(device)
        pb_sn     = player_bbox if primary_bbox is None else primary_bbox
        primary_t = torch.tensor([_to_sn(pb_sn)], dtype=torch.float32).to(device)
        stats_t   = torch.tensor([compute_region_stats(thumb)],
                                 dtype=torch.float32).to(device)
        box_t, angle_t = model(img_tensor, union_t, primary_t, stats_t)
        angle_deg = float(angle_t.squeeze().item()) * 90.0
    else:
        pb_tensor = None
        if getattr(model, 'use_player_bbox', False):
            pb_tensor = torch.tensor([_to_sn(player_bbox)],
                                     dtype=torch.float32).to(device)
        out = model(img_tensor, pb_tensor)
        if isinstance(out, tuple):
            box_t, angle_t = out
            angle_deg = float(angle_t.squeeze().item()) * 90.0
        else:
            box_t     = out
            angle_deg = 0.0

    x1, y1, x2, y2 = box_t.squeeze(0).cpu().tolist()
    x1, y1, x2, y2 = _developed_image_box([x1, y1, x2, y2], flip)
    x1 = max(0.0, x1 - 0.02)
    y1 = max(0.0, y1 - 0.02)
    x2 = min(1.0, x2 + 0.02)
    y2 = min(1.0, y2 + 0.02)

    W, H = img_size if img_size is not None else DEVELOP_SIZE
    x = int(x1 * W)
    y = int(y1 * H)
    w = max(1, int((x2 - x1) * W))
    h = max(1, int((y2 - y1) * H))
    return [x, y, w, h], angle_deg


@torch.no_grad()
def apply_color_param(model, img: Image.Image, device) -> tuple[Image.Image, list]:
    """Returns (corrected_image, param_vec)."""
    resized = img.resize(getattr(model, '_input_size', COLOR_SIZE), Image.LANCZOS)
    inp     = transforms.ToTensor()(resized).unsqueeze(0).to(device)
    vec     = model(inp).squeeze(0).cpu().tolist()
    return apply_param_vec(img, vec), vec


@torch.no_grad()
def apply_color_unet(model, img: Image.Image, device) -> Image.Image:
    orig_size = img.size
    resized   = img.resize(COLOR_SIZE, Image.LANCZOS)
    inp       = transforms.ToTensor()(resized).unsqueeze(0).to(device)
    out       = model(inp).squeeze(0).cpu()
    out_pil   = transforms.ToPILImage()(out.clamp(0, 1))
    return out_pil.resize(orig_size, Image.LANCZOS)


@torch.no_grad()
def judge_score(model, img: Image.Image, device) -> float:
    if model is None:
        return float("nan")
    resized = img.resize(THUMB_SIZE, Image.LANCZOS)
    inp     = _IMAGENET_TF(resized).unsqueeze(0).to(device)
    return float(torch.sigmoid(model(inp)).item())


# ── Apply crop ─────────────────────────────────────────────────────────────────

def apply_crop(img: Image.Image, crop_box: list, angle_deg: float) -> Image.Image:
    import math
    x, y, w, h = crop_box

    if abs(angle_deg) > 0.5:
        W, H = img.size
        rotated = img.rotate(-angle_deg, expand=True, resample=Image.BICUBIC)
        W2, H2 = rotated.size

        # PIL rotate(-angle_deg) = clockwise rotation by angle_deg.
        theta = math.radians(angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        def _xform(px, py):
            cx, cy = px - W / 2, py - H / 2
            return (cx * cos_t + cy * sin_t + W2 / 2,
                    -cx * sin_t + cy * cos_t + H2 / 2)

        # PIL.rotate(expand=True) fills the new canvas corners with black.
        # Compute the safe inscribed rectangle (no black corners) by finding
        # where the original image corners land in rotated space.
        # Order: TL, TR, BR, BL
        oc = [_xform(px, py) for px, py in [(0, 0), (W, 0), (W, H), (0, H)]]
        safe_x1 = math.ceil(max(oc[0][0], oc[3][0]))   # rightmost of TL.x, BL.x
        safe_y1 = math.ceil(max(oc[0][1], oc[1][1]))   # bottom-most of TL.y, TR.y
        safe_x2 = math.floor(min(oc[1][0], oc[2][0]))  # leftmost of TR.x, BR.x
        safe_y2 = math.floor(min(oc[3][1], oc[2][1]))  # top-most of BL.y, BR.y

        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        xs, ys = zip(*[_xform(px, py) for px, py in corners])
        x1 = max(safe_x1, int(min(xs)))
        y1 = max(safe_y1, int(min(ys)))
        x2 = min(safe_x2, int(max(xs)))
        y2 = min(safe_y2, int(max(ys)))
        return rotated.crop((x1, y1, x2, y2))

    x = max(0, min(x, img.width  - 1))
    y = max(0, min(y, img.height - 1))
    w = max(1, min(w, img.width  - x))
    h = max(1, min(h, img.height - y))
    return img.crop((x, y, x + w, y + h))


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    color_model: str = "param",
    dry_run: bool = False,
    jpeg_quality: int = 95,
) -> dict:
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    raw_files  = sorted(input_dir.glob("*.CR3")) + sorted(input_dir.glob("*.cr3"))

    if not raw_files:
        raise ValueError(f"No CR3 files found in {input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Files: {len(raw_files)}")

    cull_model, threshold = _load_cull(device)
    crop_model = _load_crop(device)
    yolo_model = _load_yolo() if isinstance(crop_model, CombinedDINOv2) else None
    color_mod  = _load_color_param(device) if color_model == "param" else _load_color_unet(device)
    judge      = _load_judge(device)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    culled = kept = 0

    for raw_path in tqdm(raw_files, desc="Processing"):
        keep, conf = predict_cull(cull_model, threshold, raw_path, device)
        if not keep:
            culled += 1
            results.append({"file": raw_path.name, "culled": True, "confidence": conf})
            continue

        kept += 1
        if yolo_model is not None:
            thumb = extract_thumbnail_ar(raw_path, max_size=512)
            union_bbox, primary_bbox = _detect_players(yolo_model, thumb)
        else:
            union_bbox = primary_bbox = None
        crop_box, angle_deg = predict_crop(crop_model, raw_path, device,
                                           player_bbox=union_bbox,
                                           primary_bbox=primary_bbox)
        raw_img = develop_raw(raw_path, size=DEVELOP_SIZE)
        cropped = apply_crop(raw_img, crop_box, angle_deg)

        if color_model == "param":
            corrected, vec = apply_color_param(color_mod, cropped, device)
        else:
            corrected = apply_color_unet(color_mod, cropped, device)
            vec       = None

        j_score = judge_score(judge, corrected, device)
        entry   = {
            "file":       raw_path.name,
            "culled":     False,
            "cull_conf":  conf,
            "crop_box":   crop_box,
            "angle_deg":  angle_deg,
            "color_params": vec,
            "judge_score": j_score,
        }

        if not dry_run:
            out_path = output_dir / (raw_path.stem + ".jpg")
            corrected.save(out_path, format="JPEG", quality=jpeg_quality)
            entry["output"] = str(out_path)

        results.append(entry)

    summary = {
        "total": len(raw_files),
        "culled": culled,
        "kept": kept,
        "cull_rate": culled / len(raw_files),
        "results": results,
    }
    if not dry_run:
        with open(output_dir / "pipeline_results.json", "w") as fh:
            json.dump(summary, fh, indent=2)

    print(f"\nDone.  Kept: {kept}  |  Culled: {culled}  |  Output: {output_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--color-model", choices=["param", "unet"], default="param")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--quality",     type=int, default=95)
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.color_model, args.dry_run, args.quality)

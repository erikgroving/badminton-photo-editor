"""
Structured pipeline runner — two-phase execution.

Phase 1  run_culling_stage()    → culling + player coverage  (stages 0-1)
Phase 2  run_processing_stage() → cropping + color correction (stages 2-3)

Output layout:
  <output_dir>/
    culls/
      passed/           <- CR3 copies (Windows) or symlinks (Mac)
      culled/
      rankings.json     <- {selection_target, threshold_used, input_dir, photos: [...]}
    crops/              <- cropped JPEGs (passed photos only)
    colors_and_lightning/ <- color-corrected JPEGs

Usage (CLI):
    python -m inference.run --input /photos/session/ --output /photos/session_output/
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHECKPOINTS_DIR, COLOR_PARAM_CKPT, COLOR_SIZE, CROP_CKPT, CULL_MODEL_NAME,
    DEVELOP_SIZE, THUMB_SIZE,
)
from data.raw_reader import develop_raw, extract_thumbnail, extract_thumbnail_ar, get_raw_flip
from inference.apply_params import apply_param_vec
from inference.pipeline import (
    _load_color_param, _load_crop, _load_cull, _load_yolo, _detect_players,
    apply_color_param, apply_crop, predict_crop, _developed_image_box,
)
from models.culling.model import build_model as build_cull_model

ProgressCb = Callable[[int, int, int, str], None]   # (stage, done, total, msg)


# ─── Player-region crop for color model ────────────────────────────────────────

def _load_yolo_detector():
    from ultralytics import YOLO
    bundled = CHECKPOINTS_DIR / "yolo11n.pt"
    weights = str(bundled) if bundled.exists() else "yolo11n.pt"
    return YOLO(weights)


def _player_region_crop(
    img: Image.Image,
    detector,
    target_size: tuple[int, int],
    pad: float = 0.15,
    min_fraction: float = 0.01,
) -> Image.Image:
    """
    Detect persons in img, crop to their union bbox (+ padding), resize to target_size.
    Falls back to center-biased crop if no detections.
    """
    W, H = img.size
    min_area = W * H * min_fraction
    try:
        results = detector(img, classes=[0], verbose=False)
        boxes = [
            (x1, y1, x2, y2)
            for r in results
            for (x1, y1, x2, y2) in (map(float, b) for b in r.boxes.xyxy.cpu().tolist())
            if (x2 - x1) * (y2 - y1) >= min_area
        ]
    except Exception:
        boxes = []

    if boxes:
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        px = (x2 - x1) * pad
        py = (y2 - y1) * pad
        x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
        x2 = min(W, x2 + px);  y2 = min(H, y2 + py)
    else:
        # Center 70% crop as fallback
        margin_x, margin_y = W * 0.15, H * 0.15
        x1, y1, x2, y2 = margin_x, margin_y, W - margin_x, H - margin_y

    return img.crop((int(x1), int(y1), int(x2), int(y2))).resize(target_size, Image.LANCZOS)


def _detect_player_bboxes(
    img: Image.Image,
    detector,
    min_fraction: float = 0.01,
) -> tuple[list | None, list | None]:
    """Return (union_bbox, primary_bbox) normalized [x1,y1,x2,y2], or (None, None)."""
    if detector is None:
        return None, None
    W, H = img.size
    min_area = W * H * min_fraction
    try:
        results = detector(img, classes=[0], verbose=False)
        boxes = [
            (x1, y1, x2, y2)
            for r in results
            for (x1, y1, x2, y2) in (map(float, b) for b in r.boxes.xyxy.cpu().tolist())
            if (x2 - x1) * (y2 - y1) >= min_area
        ]
    except Exception:
        return None, None
    if not boxes:
        return None, None
    primary = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
    union = (
        [min(b[0] for b in boxes) / W, min(b[1] for b in boxes) / H,
         max(b[2] for b in boxes) / W, max(b[3] for b in boxes) / H]
        if len(boxes) > 1 else
        [primary[0]/W, primary[1]/H, primary[2]/W, primary[3]/H]
    )
    primary_n = [primary[0]/W, primary[1]/H, primary[2]/W, primary[3]/H]
    return union, primary_n


# ─── Threshold selection ────────────────────────────────────────────────────────

def _threshold_for_selection(scores: list[float], selection_target: float) -> float:
    if not scores:
        return 0.5
    n_pass = max(1, round(len(scores) * selection_target))
    sorted_desc = sorted(scores, reverse=True)
    return float(sorted_desc[min(n_pass - 1, len(sorted_desc) - 1)])


# ─── File linking ───────────────────────────────────────────────────────────────

def _link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except (OSError, NotImplementedError):
        import shutil
        shutil.copy2(src, dst)


# ─── Image transform ────────────────────────────────────────────────────────────

_IMAGENET_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def _to_tensor(img: Image.Image, device) -> torch.Tensor:
    return _IMAGENET_TF(img).unsqueeze(0).to(device)


# ─── Phase 1: Culling + player coverage ────────────────────────────────────────

def run_culling_stage(
    input_dir:        str | Path,
    output_dir:       str | Path,
    selection_target: float = 0.20,
    player_coverage:  bool  = True,
    progress_cb:      Optional[ProgressCb] = None,
) -> dict:
    """
    Stage 0: score all photos, apply selection target.
    Stage 1: player coverage guarantee (promote one photo per uncovered cluster).
    Writes culls/passed/, culls/culled/, culls/rankings.json.
    Returns summary dict.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    raw_files  = sorted({p for p in input_dir.glob("*") if p.suffix.lower() == ".cr3"},
                        key=lambda p: p.name)

    if not raw_files:
        raise ValueError(f"No CR3 files found in {input_dir}")

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device: {device}  |  Files: {len(raw_files)}")

    cull_model, _ckpt = _load_cull_with_ckpt(device)
    cull_sz = getattr(cull_model, '_input_size', THUMB_SIZE[0])

    import shutil
    for sub in ("culls/passed", "culls/culled", "crops", "colors_and_lightning"):
        d = output_dir / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # Clear burst review state so a fresh pipeline run always starts review at burst 1
    state_file = output_dir / "burst_review_state.json"
    if state_file.exists():
        state_file.unlink()

    # ── Stage 0: score every photo ────────────────────────────────────────────
    n_total    = len(raw_files)
    raw_scores: list[tuple[Path, float]] = []

    for i, raw in enumerate(raw_files):
        if progress_cb:
            progress_cb(0, i, n_total, raw.name)
        img  = extract_thumbnail(raw, size=THUMB_SIZE)
        if cull_sz != THUMB_SIZE[0]:
            img = img.resize((cull_sz, cull_sz), Image.LANCZOS)
        with torch.no_grad():
            prob = float(torch.sigmoid(cull_model(_to_tensor(img, device))).item())
        raw_scores.append((raw, prob))

    if progress_cb:
        progress_cb(0, n_total, n_total, "Scoring complete — applying threshold...")

    raw_scores.sort(key=lambda x: x[1], reverse=True)
    score_vals = [s for _, s in raw_scores]
    threshold  = _threshold_for_selection(score_vals, selection_target)
    print(f"Selection target: {selection_target:.0%}  ->  threshold: {threshold:.3f}")

    rankings: list[dict] = []
    for rank, (raw, score) in enumerate(raw_scores, 1):
        decision = "passed" if score >= threshold else "culled"
        rankings.append({
            "filename": raw.name,
            "score":    round(score, 6),
            "rank":     rank,
            "decision": decision,
        })

    n_passed_initial = sum(1 for r in rankings if r["decision"] == "passed")
    if progress_cb:
        progress_cb(0, n_total, n_total,
                    f"Culling done: {n_passed_initial:,} passed / {n_total - n_passed_initial:,} culled")

    # ── Stage 1: player coverage ──────────────────────────────────────────────
    coverage_stats: dict = {}
    if player_coverage:
        try:
            from inference.player_coverage import run_coverage_guarantee

            def _cov_cb(frac: float, msg: str):
                if progress_cb:
                    progress_cb(1, int(frac * n_total), n_total, msg)

            rankings, coverage_stats = run_coverage_guarantee(
                photos=rankings,
                raw_dir=input_dir,
                thumb_size=THUMB_SIZE,
                device=device,
                progress_cb=_cov_cb,
            )
            n_promoted = coverage_stats.get("n_promoted", 0)
            if n_promoted:
                print(f"Player coverage: promoted {n_promoted} photo(s)")
        except Exception as exc:
            err_msg = f"Player coverage error ({type(exc).__name__}): {exc}"
            print(err_msg)
            traceback.print_exc()
            if progress_cb:
                progress_cb(1, n_total, n_total, err_msg)

    # ── Write files ───────────────────────────────────────────────────────────
    for r in rankings:
        raw_path = input_dir / r["filename"]
        _link_or_copy(raw_path, output_dir / "culls" / r["decision"] / r["filename"])

    (output_dir / "culls" / "rankings.json").write_text(json.dumps({
        "model_version":    "1.0",
        "selection_target": selection_target,
        "threshold_used":   round(threshold, 6),
        "input_dir":        str(input_dir),
        "coverage_stats":   coverage_stats,
        "photos":           rankings,
    }, indent=2))

    n_passed = sum(1 for r in rankings if r["decision"] == "passed")
    n_culled = n_total - n_passed
    print(f"Culling: {n_passed} passed  |  {n_culled} culled")

    return {
        "total":            n_total,
        "passed":           n_passed,
        "passed_cull":      n_passed_initial,
        "culled":           n_culled,
        "selection_target": selection_target,
        "threshold":        threshold,
        "coverage_stats":   coverage_stats,
        "output_dir":       str(output_dir),
    }


# ─── Phase 2: Cropping + color correction ──────────────────────────────────────

def run_processing_stage(
    output_dir:   str | Path,
    jpeg_quality: int = 95,
    progress_cb:  Optional[ProgressCb] = None,
) -> dict:
    """
    Stage 2: crop every photo in culls/passed/.
    Stage 3: color-correct every crop.
    Reads input_dir from culls/rankings.json so the caller doesn't need it.
    Returns summary dict.
    """
    output_dir     = Path(output_dir)
    rankings_path  = output_dir / "culls" / "rankings.json"
    if not rankings_path.exists():
        raise FileNotFoundError(f"No rankings.json at {rankings_path}. Run culling first.")

    data      = json.loads(rankings_path.read_text())
    input_dir = Path(data.get("input_dir", ""))

    # Use the actual filesystem state of passed/ so burst-review moves are respected
    passed_dir = output_dir / "culls" / "passed"
    _seen: set[str] = set()
    passed_raws: list[Path] = []
    for p in sorted(passed_dir.glob("*.CR3")) + sorted(passed_dir.glob("*.cr3")):
        if p.name.lower() not in _seen:
            _seen.add(p.name.lower())
            passed_raws.append(p)
    n_passed = len(passed_raws)

    if not passed_raws:
        raise ValueError("No photos in culls/passed/. Run culling (or review) first.")

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )

    crop_model  = _load_crop(device)
    color_model = _load_color_param(device)
    try:
        # One shared YOLO instance — used for both crop conditioning and color player-crop
        _detector = _load_yolo()
    except Exception as exc:
        print(f"YOLO detector unavailable ({type(exc).__name__}): {exc}. Using center-crop fallback.")
        _detector = None

    import shutil
    for sub in ("crops", "colors_and_lightning"):
        d = output_dir / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # ── Stage 2: cropping ─────────────────────────────────────────────────────
    for i, p in enumerate(passed_raws):
        if progress_cb:
            progress_cb(2, i, n_passed, p.name)
        raw_img = develop_raw(p, size=DEVELOP_SIZE, neutral=False)

        # Detect on the sensor-native (landscape) thumbnail — same space the model
        # was trained on.  Convert to developed-image space so predict_crop can
        # invert back to SN internally.
        flip = get_raw_flip(p)
        if _detector is not None:
            thumb_sn = extract_thumbnail_ar(p, max_size=512)
            union_sn, primary_sn = _detect_players(_detector, thumb_sn)
            union_bbox   = _developed_image_box(union_sn,   flip)
            primary_bbox = _developed_image_box(primary_sn, flip)
        else:
            union_bbox = primary_bbox = None
        crop_box, model_angle = predict_crop(
            crop_model, p, device,
            img_size=raw_img.size,
            player_bbox=union_bbox,
            primary_bbox=primary_bbox,
        )

        is_portrait = flip in (5, 6, 3)
        angle_deg = 0.0 if (is_portrait or abs(model_angle) > 30.0) else model_angle

        cropped = apply_crop(raw_img, crop_box, angle_deg)
        out                 = output_dir / "crops" / f"{p.stem}.jpg"
        cropped.save(out, format="JPEG", quality=jpeg_quality)

    if progress_cb:
        progress_cb(2, n_passed, n_passed, "Cropping complete")

    # ── Stage 3: color correction ─────────────────────────────────────────────
    crops_dir = output_dir / "crops"
    color_sz  = getattr(color_model, '_input_size', COLOR_SIZE)

    for i, p in enumerate(passed_raws):
        if progress_cb:
            progress_cb(3, i, n_passed, p.name)
        crop_jpg = crops_dir / f"{p.stem}.jpg"
        if not crop_jpg.exists():
            continue
        cropped = Image.open(crop_jpg).convert("RGB")
        # Feed only the player region to the color model so bright backgrounds
        # don't skew the WB/exposure prediction.
        color_inp = _player_region_crop(cropped, _detector, color_sz)
        inp = transforms.ToTensor()(color_inp).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = color_model(inp).squeeze(0).cpu().tolist()
        corrected = apply_param_vec(cropped, vec)
        out = output_dir / "colors_and_lightning" / f"{p.stem}.jpg"
        corrected.save(out, format="JPEG", quality=jpeg_quality)

    if progress_cb:
        progress_cb(3, n_passed, n_passed, "Color correction complete")

    summary = {
        "processed":       n_passed,   # key read by app/main.py _on_proc_finished
        "total_processed": n_passed,   # kept for backward-compat / CLI callers
        "output_dir":      str(output_dir),
    }
    (output_dir / "processing_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone.  Output: {output_dir}")
    return summary


# ─── Combined runner (backward-compat / CLI) ────────────────────────────────────

def run_pipeline_structured(
    input_dir:        str | Path,
    output_dir:       str | Path,
    selection_target: float = 0.20,
    jpeg_quality:     int   = 95,
    player_coverage:  bool  = True,
    progress_cb:      Optional[ProgressCb] = None,
) -> dict:
    cull_summary = run_culling_stage(
        input_dir, output_dir,
        selection_target=selection_target,
        player_coverage=player_coverage,
        progress_cb=progress_cb,
    )
    proc_summary = run_processing_stage(
        output_dir,
        jpeg_quality=jpeg_quality,
        progress_cb=progress_cb,
    )
    return {**cull_summary, **proc_summary}


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _load_cull_with_ckpt(device) -> tuple:
    ckpt_files = list(CHECKPOINTS_DIR.glob("culling_*.pt"))
    if not ckpt_files:
        raise FileNotFoundError("No culling checkpoint found. Train first.")
    best = None
    for f in ckpt_files:
        d    = torch.load(f, map_location="cpu")
        cost = d.get("metrics", {}).get("asym_cost", float("inf"))
        if best is None or cost < best[1]:
            best = (f, cost, d)
    ckpt             = best[2]
    backbone         = ckpt.get("backbone", CULL_MODEL_NAME)
    dynamic_img_size = ckpt.get("dynamic_img_size", False)
    model    = build_cull_model(backbone=backbone, pretrained=False,
                                dynamic_img_size=dynamic_img_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    input_size = ckpt.get("input_size")
    if input_size is None:
        is_vit = backbone and ("vit" in backbone or "swin" in backbone)
        input_size = 518 if is_vit else THUMB_SIZE[0]
    model._input_size = input_size
    return model, ckpt


# ─── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--selection",   type=float, default=0.20)
    parser.add_argument("--quality",     type=int,   default=95)
    parser.add_argument("--no-coverage", action="store_true")
    args = parser.parse_args()
    run_pipeline_structured(
        args.input, args.output,
        selection_target=args.selection,
        jpeg_quality=args.quality,
        player_coverage=not args.no_coverage,
    )

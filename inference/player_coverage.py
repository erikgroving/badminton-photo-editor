"""
Player coverage guarantee — post-processing step after culling.

After the main cull, this module:
  1. Detects the largest player per photo (YOLOv8-nano)                  → 0-40 %
  2. Embeds via DINOv2 (768d)                                             → 40-70 %
  3. Look-back/look-ahead boundary detection → segments                   → 70-80 %
  4. Promotes best photo from each uncovered segment                      → 80-100 %

Design rationale:
  Look-back/look-ahead boundary detection: at each position i, compare the
  average embedding of the previous K=20 photos vs the next K=20 photos.
  Intra-player oscillations cancel in both windows (no false boundaries).
  True player transitions produce clearly different averages (real boundary).
  K can be large without blurring real boundaries — unlike sliding-window
  smoothing which bleeds adjacent players into each other at transition points.
  Result: ~21 segments for 5 players (Demo), all player boundaries clean.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ── Constants ──────────────────────────────────────────────────────────────────

_BOUNDARY_K        = 20     # look-back / look-ahead window size for boundary detection
_CHANGE_THRESHOLD  = 0.25   # distance between look-back avg and look-ahead avg → new segment
_MIN_CROP_FRACTION = 0.01

# Kept for API compatibility with test scripts
_SMOOTH_K          = 3
_CLUSTER_DISTANCE  = 0.65
_SEQ_K             = 20
_SIM_THRESH        = 0.65
_COLOR_ALPHA       = 0.2
_MERGE_DISTANCE    = 0.45
_UPPER_BODY_FRAC   = 0.60

_IMAGENET_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _file_num(fname: str) -> int:
    m = re.search(r'(\d{4,})', Path(fname).stem)
    return int(m.group(1)) if m else 0


# ── Model builders ─────────────────────────────────────────────────────────────

def build_detector():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import CHECKPOINTS_DIR
    from ultralytics import YOLO
    bundled = CHECKPOINTS_DIR / 'yolo11n.pt'
    weights = str(bundled) if bundled.exists() else 'yolo11n.pt'
    return YOLO(weights)


def build_embedder(device: torch.device) -> torch.nn.Module:
    import timm
    model = timm.create_model(
        'vit_base_patch14_reg4_dinov2',
        pretrained=True,
        num_classes=0,
        global_pool='token',
        dynamic_img_size=True,
    )
    return model.eval().to(device)


# ── Detection ──────────────────────────────────────────────────────────────────

def detect_player_crops(
    img: Image.Image,
    detector,
    min_crop_fraction: float = _MIN_CROP_FRACTION,
) -> list[Image.Image]:
    """Return all detected person crops (kept for test-script compatibility)."""
    results = detector(img, classes=[0], verbose=False)
    crops = []
    w, h = img.size
    min_area = w * h * min_crop_fraction
    for r in results:
        for box in r.boxes.xyxy.cpu().tolist():
            x1, y1, x2, y2 = map(int, box)
            if (x2 - x1) * (y2 - y1) < min_area:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
    return crops


def _detect_largest_box(
    img: Image.Image,
    detector,
    min_crop_fraction: float = _MIN_CROP_FRACTION,
) -> Optional[tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) of the largest detected person, or None."""
    results = detector(img, classes=[0], verbose=False)
    w, h = img.size
    min_area = w * h * min_crop_fraction
    best_box, best_area = None, 0.0
    for r in results:
        for box in r.boxes.xyxy.cpu().tolist():
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area and area > best_area:
                best_box, best_area = (x1, y1, x2, y2), area
    return best_box


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_crops(crops: list[Image.Image], embedder, device: torch.device) -> np.ndarray:
    """Kept for test-script compatibility."""
    if not crops:
        return np.zeros((0, 768), dtype=np.float32)
    batch = torch.stack([_IMAGENET_TF(c) for c in crops]).to(device)
    batch = F.interpolate(batch, size=(518, 518), mode='bilinear', align_corners=False)
    with torch.no_grad():
        feats = embedder(batch)
        feats = F.normalize(feats, dim=1)
    return feats.cpu().float().numpy()


# ── Clustering helpers (kept for test-script compatibility) ───────────────────

def cluster_embeddings(
    embeddings: np.ndarray,
    distance_threshold: float = _CLUSTER_DISTANCE,
) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    if len(embeddings) == 0:
        return np.array([], dtype=int)
    if len(embeddings) == 1:
        return np.array([0], dtype=int)
    return AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average',
    ).fit_predict(embeddings)


# ── SeqKNN (kept for test-script / research use) ─────────────────────────────

def _seqknn_components(
    embs_normed: np.ndarray,
    K: int = _SEQ_K,
    sim_thresh: float = _SIM_THRESH,
) -> np.ndarray:
    """Sequential K-NN connected-components (used in research; kept for scripts)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(embs_normed)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    rows, cols = [], []
    for i in range(n):
        lo = max(0, i - K)
        hi = min(n, i + K + 1)
        sims = embs_normed[lo:hi] @ embs_normed[i]
        for offset, sim in enumerate(sims):
            j = lo + offset
            if j != i and float(sim) >= sim_thresh:
                rows.append(i)
                cols.append(j)

    if not rows:
        return np.arange(n, dtype=int)

    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(adj, directed=False, connection='weak')
    return labels


def _color_histogram(
    crop: Image.Image,
    n_h: int = 32,
    n_s: int = 16,
    n_v: int = 16,
) -> np.ndarray:
    """L2-normalised HSV histogram (kept for research scripts)."""
    try:
        import cv2
        arr = cv2.cvtColor(np.array(crop.convert('RGB')), cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist(
            [arr], [0, 1, 2], None,
            [n_h, n_s, n_v],
            [0, 180, 0, 256, 0, 256],
        ).flatten().astype(np.float32)
    except Exception:
        arr = np.array(crop.convert('RGB').resize((8, 8))).flatten().astype(np.float32)
        hist = arr
    norm = np.linalg.norm(hist)
    return hist / (norm + 1e-8)


# ── Temporal segmentation ─────────────────────────────────────────────────────

def _temporal_segments(
    fnames_sorted: list[str],
    emb_by_fname: dict[str, np.ndarray],
    boundary_k: int = _BOUNDARY_K,
    change_threshold: float = _CHANGE_THRESHOLD,
) -> list[list[str]]:
    """
    Group sorted filenames into temporal segments using look-back / look-ahead
    boundary detection.

    At each position i, the cosine distance between the average of the
    previous K embeddings and the average of the next K embeddings is
    computed. A new segment starts when this distance exceeds change_threshold.

    Advantages over simple smoothing + running-average changepoints:
    - Oscillations within one player's run cancel in both windows → no false breaks
    - True player transitions produce clearly different look-back and look-ahead
      averages regardless of K size → no boundary blurring
    - Large K can be used freely for noise suppression without blurring boundaries
    """
    detected = [(i, fn) for i, fn in enumerate(fnames_sorted) if fn in emb_by_fname]
    if not detected:
        return []

    det_embs = np.stack([emb_by_fname[fn] for _, fn in detected])
    m = len(det_embs)

    # Find boundary positions
    boundary_positions = [0]
    for i in range(1, m):
        back  = det_embs[max(0, i - boundary_k):i]
        ahead = det_embs[i:min(m, i + boundary_k)]
        b_avg = back.mean(axis=0);  b_avg /= np.linalg.norm(b_avg) + 1e-8
        a_avg = ahead.mean(axis=0); a_avg /= np.linalg.norm(a_avg) + 1e-8
        if 1.0 - float(np.dot(b_avg, a_avg)) > change_threshold:
            boundary_positions.append(i)
    boundary_positions.append(m)

    segments: list[list[str]] = []
    for a, b in zip(boundary_positions, boundary_positions[1:]):
        segments.append([detected[j][1] for j in range(a, b)])

    # Attach undetected photos to the nearest detected photo's segment
    detected_positions = [orig_i for orig_i, _ in detected]
    det_to_seg: dict[int, int] = {}
    for seg_i, (a, b) in enumerate(zip(boundary_positions, boundary_positions[1:])):
        for j in range(a, b):
            det_to_seg[j] = seg_i

    for i, fn in enumerate(fnames_sorted):
        if fn in emb_by_fname:
            continue
        if not detected_positions:
            continue
        nearest_orig = min(detected_positions, key=lambda di: abs(di - i))
        nearest_det_i = next(j for j, (oi, _) in enumerate(detected) if oi == nearest_orig)
        segments[det_to_seg[nearest_det_i]].append(fn)

    return segments


# ── Main entry point ───────────────────────────────────────────────────────────

def run_coverage_guarantee(
    photos: list[dict],
    raw_dir: Path,
    thumb_size: tuple[int, int],
    device: torch.device,
    distance_threshold: float = _CLUSTER_DISTANCE,   # unused; kept for API compat
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> tuple[list[dict], dict]:
    """
    Guarantee at least one passed photo per detected player segment.

    Algorithm: smooth DINOv2 embeddings (K=5 window) then temporal changepoint
    detection (threshold=0.15). Promotes the highest-scoring photo from each
    segment that has no passed photos.

    Typical output: 10-20 segments for a 5-20 player match, vs 35+ fine clusters
    from the SeqKNN approach — giving bounded, predictable coverage promotions.

    progress_cb(fraction, message):
      0.00–0.40  YOLO detection
      0.40–0.70  DINOv2 embedding
      0.70–0.80  temporal segmentation
      0.80–1.00  promotion
    """
    from data.raw_reader import extract_thumbnail

    detector = build_detector()
    embedder = build_embedder(device)

    n = len(photos)

    # ── Phase 1: Detect largest person per photo (0 → 40 %) ──────────────────
    box_by_fname: dict[str, tuple] = {}
    for i, p in enumerate(photos):
        if progress_cb:
            progress_cb(i / n * 0.40, f"Detecting: {p['filename']}")
        raw_path = raw_dir / p['filename']
        if not raw_path.exists():
            continue
        try:
            img = extract_thumbnail(raw_path, size=thumb_size)
            box = _detect_largest_box(img, detector)
            if box is not None:
                box_by_fname[p['filename']] = (img, box)
        except Exception:
            pass

    n_photos_with_players = len(box_by_fname)
    if not box_by_fname:
        if progress_cb:
            progress_cb(1.0, "No players detected — coverage skipped")
        return photos, {'n_clusters': 0, 'n_promoted': 0, 'n_photos_with_players': 0}

    # ── Phase 2: DINOv2 embedding (40 → 70 %) ────────────────────────────────
    emb_by_fname: dict[str, np.ndarray] = {}
    fnames_with_box = list(box_by_fname.keys())
    for i, fname in enumerate(fnames_with_box):
        if progress_cb:
            progress_cb(0.40 + i / len(fnames_with_box) * 0.30, f"Embedding: {fname}")
        img, (x1, y1, x2, y2) = box_by_fname[fname]
        try:
            crop = img.crop((x1, y1, x2, y2))
            emb = embed_crops([crop], embedder, device)[0]
            emb /= np.linalg.norm(emb) + 1e-8
            emb_by_fname[fname] = emb
        except Exception:
            pass

    if progress_cb:
        progress_cb(0.70, "Temporal segmentation...")

    # ── Phase 3: Smooth + changepoint segmentation (70 → 80 %) ───────────────
    fnames_sorted = sorted([p['filename'] for p in photos], key=_file_num)
    segments = _temporal_segments(fnames_sorted, emb_by_fname,
                                  boundary_k=_BOUNDARY_K,
                                  change_threshold=_CHANGE_THRESHOLD)
    n_clusters = len(segments)

    if not segments:
        if progress_cb:
            progress_cb(1.0, "No segments found — coverage skipped")
        return photos, {'n_clusters': 0, 'n_promoted': 0, 'n_photos_with_players': 0}

    # ── Phase 4: Promote from uncovered segments (80 → 100 %) ────────────────
    score_map    = {p['filename']: p['score']    for p in photos}
    decision_map = {p['filename']: p['decision'] for p in photos}

    promote: set[str] = set()
    for seg_fnames in segments:
        if not any(decision_map.get(f) == 'passed' for f in seg_fnames):
            best = max(seg_fnames, key=lambda f: score_map.get(f, 0.0))
            promote.add(best)

    updated = []
    for p in photos:
        if p['filename'] in promote and p['decision'] == 'culled':
            updated.append({**p, 'decision': 'passed', 'coverage_promoted': True})
        else:
            updated.append(p)

    stats = {
        'n_clusters':            n_clusters,
        'n_promoted':            len(promote),
        'n_photos_with_players': n_photos_with_players,
    }
    if progress_cb:
        progress_cb(
            1.0,
            f"Coverage done: {len(promote)} photo(s) promoted across {n_clusters} segments",
        )
    return updated, stats

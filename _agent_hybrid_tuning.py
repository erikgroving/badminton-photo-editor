"""
Hybrid OSNet+DINOv2 player-ID tuning script.

Finds Pareto-optimal operating point for the hybrid method:
  50% OSNet + 50% PCA-whitened DINOv2 + temporal changepoint segmentation

Parameter grid (500 combos):
  alpha_values   = [0.3, 0.4, 0.5, 0.6, 0.7]
  pca_dims       = [16, 32, 64, 128]
  change_thresh  = [0.20, 0.25, 0.30, 0.35, 0.40]
  cluster_thresh = [0.40, 0.45, 0.50, 0.55, 0.60]

Efficiency:
  1. Compute OSNet embeddings once and cache to disk
  2. Compute DINOv2 embeddings once and cache to disk
  3. For each pca_dims, fit PCA once and cache whitened embeddings
  4. Fast inner loop over (alpha, change_thresh, cluster_thresh)
"""
from __future__ import annotations

import re
import sys
import time
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from torchvision import transforms

sys.path.insert(0, r'C:\Users\erikg\OneDrive\Desktop\CodePlayground\badminton-photo-editor')
from inference.player_coverage import build_detector, build_embedder, embed_crops
from data.raw_reader import extract_thumbnail

# ── Ground truth ───────────────────────────────────────────────────────────────

RANGES = [
    (1,    78,   1), (79,   107,  2), (108,  126,  3), (127,  150,  2),
    (151,  177,  4), (178,  186,  5), (187,  194,  6), (195,  227,  5),
    (228,  252,  6), (253,  290,  7), (291,  296,  8), (297,  327,  9),
    (328,  347,  10), (348,  401,  11), (402,  442,  12), (443,  473,  13),
    (474,  485,  14), (486,  509,  15), (510,  527,  14), (528,  575,  16),
    (576,  595,  17), (596,  634,  18), (635,  653,  19), (654,  696,  20),
]
N_TRUE_WA = 20

DEMO_GROUPS = {
    1: set(range(1316, 1344)),   # 1316-1343
    2: set(range(1344, 1390)),   # 1344-1389
    3: set(range(1390, 1447)),   # 1390-1446
    4: set(range(1447, 1480)),   # 1447-1479
    5: set(range(1480, 1487)),   # 1480-1486
}
N_TRUE_DEMO = 5


def file_num(fname: str) -> int:
    m = re.search(r'(\d{4,})', Path(fname).stem)
    return int(m.group(1)) if m else -1


def true_player_wa(fname: str) -> int:
    n = file_num(fname)
    for s, e, pid in RANGES:
        if s <= n <= e:
            return pid
    return 0


def true_player_demo(fname: str) -> int:
    n = file_num(fname)
    for g, rng in DEMO_GROUPS.items():
        if n in rng:
            return g
    return 0


# ── OSNet embedder ─────────────────────────────────────────────────────────────

_REID_TF = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_osnet(device: torch.device) -> torch.nn.Module:
    import torchreid
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=1000,
        pretrained=True,
    )
    model.eval()

    class OSNetFeatureExtractor(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.base.classifier = torch.nn.Identity()

        def forward(self, x):
            return self.base(x)

    wrapper = OSNetFeatureExtractor(model)
    return wrapper.eval().to(device)


def embed_osnet_batch(crops: list[Image.Image], osnet, device: torch.device) -> np.ndarray:
    if not crops:
        return np.zeros((0, 512), dtype=np.float32)
    # Process in mini-batches of 32 to avoid OOM
    results = []
    batch_size = 32
    for i in range(0, len(crops), batch_size):
        batch_crops = crops[i:i + batch_size]
        batch = torch.stack([_REID_TF(c) for c in batch_crops]).to(device)
        with torch.no_grad():
            feats = osnet(batch)
            feats = F.normalize(feats, dim=1)
        results.append(feats.cpu().float().numpy())
    return np.concatenate(results, axis=0)


# ── Detect largest crop ────────────────────────────────────────────────────────

def collect_photo_data(
    raw_paths: list[Path],
    detector,
    thumb_size: tuple = (448, 448),
    tag: str = "",
) -> tuple[list[str], list[Image.Image]]:
    """Detect largest person per photo. Returns (filenames, crops)."""
    fnames = []
    crops = []
    total = len(raw_paths)
    t0 = time.time()
    for i, p in enumerate(raw_paths):
        try:
            img = extract_thumbnail(p, size=thumb_size)
            results = detector(img, classes=[0], verbose=False)
            w, h = img.size
            min_area = w * h * 0.01
            boxes = []
            for r in results:
                for box in r.boxes.xyxy.cpu().tolist():
                    x1, y1, x2, y2 = map(int, box)
                    if (x2 - x1) * (y2 - y1) >= min_area:
                        boxes.append((x1, y1, x2, y2))
            if not boxes:
                continue
            largest = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            crops.append(img.crop(largest))
            fnames.append(p.name)
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"  {tag} {i+1}/{total} ({len(crops)} detected)  ETA {eta:.0f}s")
    print(f"  {tag} Done: {len(crops)}/{total} photos with detections")
    return fnames, crops


# ── Temporal segmentation + cluster ───────────────────────────────────────────

def temporal_cluster(
    fnames: list[str],
    emb_matrix: np.ndarray,    # (N, D) — already L2-normalized
    change_thresh: float,
    cluster_thresh: float,
) -> tuple[np.ndarray, int]:
    """
    Temporal changepoint segmentation followed by agglomerative clustering.
    Returns (photo_labels, n_clusters) where photo_labels[i] is cluster index for fnames[i].
    """
    # Sort by file number
    order = sorted(range(len(fnames)), key=lambda i: file_num(fnames[i]))
    sorted_fnames = [fnames[i] for i in order]
    sorted_embs = emb_matrix[order]  # already normalized

    # Build segments
    segments: list[tuple[list[int], np.ndarray]] = []  # (photo_indices_in_sorted, avg_emb)
    seg_indices = [0]
    seg_embs_list = [sorted_embs[0].copy()]
    seg_avg = sorted_embs[0].copy()

    for i in range(1, len(sorted_fnames)):
        e = sorted_embs[i]
        dist = 1.0 - float(np.dot(seg_avg, e))
        if dist > change_thresh:
            avg = np.mean(seg_embs_list, axis=0)
            norm = np.linalg.norm(avg) + 1e-8
            segments.append((seg_indices, avg / norm))
            seg_indices = [i]
            seg_embs_list = [e.copy()]
            seg_avg = e.copy()
        else:
            seg_indices = seg_indices + [i]
            seg_embs_list.append(e.copy())
            avg = np.mean(seg_embs_list, axis=0)
            seg_avg = avg / (np.linalg.norm(avg) + 1e-8)

    # Flush last segment
    avg = np.mean(seg_embs_list, axis=0)
    segments.append((seg_indices, avg / (np.linalg.norm(avg) + 1e-8)))

    # Cluster segment embeddings
    seg_matrix = np.stack([s[1] for s in segments])
    n_segs = len(segments)

    if n_segs == 1:
        seg_labels = np.array([0])
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cluster_thresh,
            metric='cosine',
            linkage='average',
        )
        seg_labels = clustering.fit_predict(seg_matrix)

    n_clusters = int(seg_labels.max()) + 1

    # Map back to photos (in sorted order)
    photo_labels_sorted = np.zeros(len(sorted_fnames), dtype=int)
    for seg_idx, (photo_indices, _) in enumerate(segments):
        cid = int(seg_labels[seg_idx])
        for pi in photo_indices:
            photo_labels_sorted[pi] = cid

    # Reorder back to original order
    photo_labels = np.zeros(len(fnames), dtype=int)
    for sorted_pos, orig_pos in enumerate(order):
        photo_labels[orig_pos] = photo_labels_sorted[sorted_pos]

    return photo_labels, n_clusters


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    labels: np.ndarray,
    fnames: list[str],
    true_player_fn,
    n_true: int,
) -> dict:
    gt = [true_player_fn(f) for f in fnames]
    n_clusters = int(labels.max()) + 1

    cluster_to_players: dict[int, set] = {}
    player_to_clusters: dict[int, set] = {}
    for i, fname in enumerate(fnames):
        pid = gt[i]
        cid = int(labels[i])
        cluster_to_players.setdefault(cid, set()).add(pid)
        player_to_clusters.setdefault(pid, set()).add(cid)

    covered = set(player_to_clusters.keys()) - {0}
    pure = sum(1 for ps in cluster_to_players.values() if len(ps) == 1)
    purity = pure / n_clusters if n_clusters > 0 else 0.0
    coverage_frac = len(covered) / n_true

    # ARI on labeled samples only
    labeled_mask = [p != 0 for p in gt]
    gt_arr = np.array([gt[i] for i, m in enumerate(labeled_mask) if m])
    pred_arr = np.array([int(labels[i]) for i, m in enumerate(labeled_mask) if m])
    ari = adjusted_rand_score(gt_arr, pred_arr) if len(gt_arr) > 1 else 0.0

    return {
        'n_clusters': n_clusters,
        'covered': len(covered),
        'n_true': n_true,
        'coverage_frac': coverage_frac,
        'pure': pure,
        'purity': purity,
        'composite': coverage_frac * purity,
        'ari': ari,
    }


# ── PCA whitening ──────────────────────────────────────────────────────────────

def pca_whiten(embs: np.ndarray, n_components: int, fit_embs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Fit PCA on fit_embs (or embs if None), transform embs.
    Returns L2-normalized whitened embeddings.
    """
    if fit_embs is None:
        fit_embs = embs
    mean = fit_embs.mean(axis=0, keepdims=True)
    centered = embs - mean
    fit_centered = fit_embs - mean
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(fit_centered)
    whitened = pca.transform(centered)
    norms = np.linalg.norm(whitened, axis=1, keepdims=True) + 1e-8
    return whitened / norms


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 70)

    CACHE_DIR = Path(r'C:\Users\erikg\OneDrive\Desktop\CodePlayground\badminton-photo-editor\cache')
    CACHE_DIR.mkdir(exist_ok=True)

    wa_osnet_cache   = CACHE_DIR / 'hybrid_wa_osnet.npy'
    wa_dinov2_cache  = CACHE_DIR / 'hybrid_wa_dinov2.npy'
    wa_fnames_cache  = CACHE_DIR / 'hybrid_wa_fnames.npy'
    demo_osnet_cache  = CACHE_DIR / 'hybrid_demo_osnet.npy'
    demo_dinov2_cache = CACHE_DIR / 'hybrid_demo_dinov2.npy'
    demo_fnames_cache = CACHE_DIR / 'hybrid_demo_fnames.npy'

    # ── Load datasets ──────────────────────────────────────────────────────────
    wa_dir   = Path(r'C:\Users\erikg\Downloads\Jay Ma Photography Training\202605 WA Open MS Raws')
    demo_dir = Path(r'C:\Users\erikg\Downloads\Demo')

    _seen: set = set()
    wa_raws = []
    for p in sorted(wa_dir.glob('*.CR3'), key=lambda x: x.name):
        if p.name.lower() in _seen:
            continue
        _seen.add(p.name.lower())
        n = file_num(p.name)
        if 1 <= n <= 696:
            wa_raws.append(p)

    _seen2: set = set()
    demo_raws = []
    for p in sorted(demo_dir.glob('*.CR3'), key=lambda x: x.name):
        if p.name.lower() not in _seen2:
            _seen2.add(p.name.lower())
            demo_raws.append(p)

    print(f"WA Open: {len(wa_raws)} photos  (20 true players)")
    print(f"Demo:    {len(demo_raws)} photos  (5 true players)")

    # ── Load or compute embeddings ─────────────────────────────────────────────

    if (wa_osnet_cache.exists() and wa_dinov2_cache.exists() and wa_fnames_cache.exists() and
            demo_osnet_cache.exists() and demo_dinov2_cache.exists() and demo_fnames_cache.exists()):
        print("\nLoading cached embeddings...")
        wa_osnet   = np.load(str(wa_osnet_cache))
        wa_dinov2  = np.load(str(wa_dinov2_cache))
        wa_fnames  = list(np.load(str(wa_fnames_cache), allow_pickle=True))
        demo_osnet  = np.load(str(demo_osnet_cache))
        demo_dinov2 = np.load(str(demo_dinov2_cache))
        demo_fnames = list(np.load(str(demo_fnames_cache), allow_pickle=True))
        print(f"  WA:   OSNet={wa_osnet.shape}, DINOv2={wa_dinov2.shape}, n={len(wa_fnames)}")
        print(f"  Demo: OSNet={demo_osnet.shape}, DINOv2={demo_dinov2.shape}, n={len(demo_fnames)}")
    else:
        print("\nBuilding models...")
        detector = build_detector()
        dinov2   = build_embedder(device)
        osnet    = build_osnet(device)
        print("  Models loaded")

        print("\nDetecting largest person per photo...")
        wa_fnames, wa_crops = collect_photo_data(wa_raws, detector, tag="WA")
        demo_fnames, demo_crops = collect_photo_data(demo_raws, detector, tag="Demo")

        print("\nComputing OSNet embeddings (512-d)...")
        wa_osnet   = embed_osnet_batch(wa_crops, osnet, device)
        demo_osnet = embed_osnet_batch(demo_crops, osnet, device)
        print(f"  WA OSNet: {wa_osnet.shape}, Demo OSNet: {demo_osnet.shape}")

        print("\nComputing DINOv2 embeddings (768-d)...")
        # Process in batches through the existing embed_crops function
        wa_dinov2_list = []
        batch_size = 16
        for i in range(0, len(wa_crops), batch_size):
            batch = wa_crops[i:i + batch_size]
            embs = embed_crops(batch, dinov2, device)
            wa_dinov2_list.append(embs)
            if (i // batch_size + 1) % 10 == 0:
                print(f"  WA DINOv2: {min(i + batch_size, len(wa_crops))}/{len(wa_crops)}")
        wa_dinov2 = np.concatenate(wa_dinov2_list, axis=0)

        demo_dinov2_list = []
        for i in range(0, len(demo_crops), batch_size):
            batch = demo_crops[i:i + batch_size]
            embs = embed_crops(batch, dinov2, device)
            demo_dinov2_list.append(embs)
        demo_dinov2 = np.concatenate(demo_dinov2_list, axis=0)
        print(f"  WA DINOv2: {wa_dinov2.shape}, Demo DINOv2: {demo_dinov2.shape}")

        # Save caches
        print("\nSaving embedding caches...")
        np.save(str(wa_osnet_cache),   wa_osnet)
        np.save(str(wa_dinov2_cache),  wa_dinov2)
        np.save(str(wa_fnames_cache),  np.array(wa_fnames, dtype=object))
        np.save(str(demo_osnet_cache),  demo_osnet)
        np.save(str(demo_dinov2_cache), demo_dinov2)
        np.save(str(demo_fnames_cache), np.array(demo_fnames, dtype=object))
        print("  Saved.")

    # ── Parameter grid ─────────────────────────────────────────────────────────

    alpha_values   = [0.3, 0.4, 0.5, 0.6, 0.7]
    pca_dims_list  = [16, 32, 64, 128]
    change_threshs = [0.20, 0.25, 0.30, 0.35, 0.40]
    cluster_threshs = [0.40, 0.45, 0.50, 0.55, 0.60]

    total_configs = len(alpha_values) * len(pca_dims_list) * len(change_threshs) * len(cluster_threshs)
    print(f"\nParameter grid: {len(alpha_values)} alpha × {len(pca_dims_list)} pca × "
          f"{len(change_threshs)} change × {len(cluster_threshs)} cluster = {total_configs} configs")

    # ── Precompute PCA-whitened DINOv2 for each pca_dims ─────────────────────

    print("\nPrecomputing PCA-whitened DINOv2 embeddings...")
    wa_whitened_by_dim: dict[int, np.ndarray] = {}
    demo_whitened_by_dim: dict[int, np.ndarray] = {}
    for n_pca in pca_dims_list:
        # Fit PCA on WA data, apply to both WA and Demo
        wa_whitened  = pca_whiten(wa_dinov2,   n_pca, fit_embs=wa_dinov2)
        demo_whitened = pca_whiten(demo_dinov2, n_pca, fit_embs=wa_dinov2)  # fit on WA
        wa_whitened_by_dim[n_pca]   = wa_whitened
        demo_whitened_by_dim[n_pca] = demo_whitened
        print(f"  PCA-{n_pca}: WA={wa_whitened.shape}, Demo={demo_whitened.shape}")

    # OSNet is already L2-normalized; ensure it
    wa_osnet_norm   = wa_osnet   / (np.linalg.norm(wa_osnet,   axis=1, keepdims=True) + 1e-8)
    demo_osnet_norm = demo_osnet / (np.linalg.norm(demo_osnet, axis=1, keepdims=True) + 1e-8)

    # ── Grid search ───────────────────────────────────────────────────────────

    print("\nRunning grid search...")
    results = []
    t0 = time.time()
    n_done = 0

    for n_pca in pca_dims_list:
        wa_white   = wa_whitened_by_dim[n_pca]
        for alpha in alpha_values:
            # Build hybrid embedding: alpha * osnet + (1-alpha) * whitened_dinov2
            # Then L2-normalize the combined vector
            wa_hybrid   = np.concatenate([alpha * wa_osnet_norm,   (1 - alpha) * wa_white],   axis=1)
            wa_hybrid  /= np.linalg.norm(wa_hybrid, axis=1, keepdims=True) + 1e-8

            for change_thresh in change_threshs:
                for cluster_thresh in cluster_threshs:
                    labels, n_clusters = temporal_cluster(
                        wa_fnames, wa_hybrid, change_thresh, cluster_thresh
                    )
                    m = compute_metrics(labels, wa_fnames, true_player_wa, N_TRUE_WA)
                    m.update({
                        'alpha': alpha,
                        'n_pca': n_pca,
                        'change_thresh': change_thresh,
                        'cluster_thresh': cluster_thresh,
                    })
                    results.append(m)
                    n_done += 1

    elapsed = time.time() - t0
    print(f"Grid search done: {n_done} configs in {elapsed:.1f}s")

    # ── Sort and display results ───────────────────────────────────────────────

    results.sort(key=lambda x: x['ari'], reverse=True)

    print("\n" + "=" * 90)
    print("=== TOP 20 CONFIGS BY ARI ===")
    print("=" * 90)
    hdr = f"{'Rank':>4}  {'alpha':>5}  {'pca':>4}  {'chg_t':>5}  {'clu_t':>5}  " \
          f"{'n_cl':>4}  {'cov':>6}  {'purity':>7}  {'composite':>9}  {'ARI':>7}"
    print(hdr)
    print("-" * 90)
    for rank, r in enumerate(results[:20], 1):
        print(
            f"{rank:>4}  {r['alpha']:>5.1f}  {r['n_pca']:>4d}  {r['change_thresh']:>5.2f}  "
            f"{r['cluster_thresh']:>5.2f}  {r['n_clusters']:>4d}  "
            f"{r['covered']:>2}/{r['n_true']}  {r['purity']:>7.3f}  "
            f"{r['composite']:>9.3f}  {r['ari']:>7.3f}"
        )

    print("\n" + "=" * 90)
    print("=== TOP 20 CONFIGS BY COMPOSITE (coverage × purity) ===")
    print("=" * 90)
    by_composite = sorted(results, key=lambda x: x['composite'], reverse=True)
    print(hdr)
    print("-" * 90)
    for rank, r in enumerate(by_composite[:20], 1):
        print(
            f"{rank:>4}  {r['alpha']:>5.1f}  {r['n_pca']:>4d}  {r['change_thresh']:>5.2f}  "
            f"{r['cluster_thresh']:>5.2f}  {r['n_clusters']:>4d}  "
            f"{r['covered']:>2}/{r['n_true']}  {r['purity']:>7.3f}  "
            f"{r['composite']:>9.3f}  {r['ari']:>7.3f}"
        )

    # ── Pareto frontier: non-dominated on (ARI ↑, n_clusters ↓ with cap 25) ──

    print("\n" + "=" * 90)
    print("=== PARETO FRONTIER: non-dominated on (ARI ↑, n_clusters ≤ 25) ===")
    print("=" * 90)

    # Filter to n_clusters <= 25 (compact), then find non-dominated on (ARI, -n_clusters)
    compact = [r for r in results if r['n_clusters'] <= 25]
    pareto = []
    for r in compact:
        dominated = False
        for other in compact:
            if other is r:
                continue
            # other dominates r if it's at least as good on both objectives and better on one
            if (other['ari'] >= r['ari'] and other['n_clusters'] <= r['n_clusters'] and
                    (other['ari'] > r['ari'] or other['n_clusters'] < r['n_clusters'])):
                dominated = True
                break
        if not dominated:
            pareto.append(r)

    pareto.sort(key=lambda x: x['ari'], reverse=True)
    print(f"Pareto-optimal configs (n_clusters ≤ 25): {len(pareto)}")
    print(hdr)
    print("-" * 90)
    for rank, r in enumerate(pareto, 1):
        print(
            f"{rank:>4}  {r['alpha']:>5.1f}  {r['n_pca']:>4d}  {r['change_thresh']:>5.2f}  "
            f"{r['cluster_thresh']:>5.2f}  {r['n_clusters']:>4d}  "
            f"{r['covered']:>2}/{r['n_true']}  {r['purity']:>7.3f}  "
            f"{r['composite']:>9.3f}  {r['ari']:>7.3f}"
        )

    # ── Key question analysis ──────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("=== PARAMETER ANALYSIS ===")
    print("=" * 70)

    # Q1: Best PCA dimensionality
    print("\nBest ARI by PCA dimension (averaged over all other params):")
    for n_pca in pca_dims_list:
        sub = [r for r in results if r['n_pca'] == n_pca]
        aris = [r['ari'] for r in sub]
        print(f"  PCA-{n_pca:3d}: mean ARI={np.mean(aris):.4f}  max ARI={np.max(aris):.4f}  "
              f"top5_mean={np.mean(sorted(aris, reverse=True)[:5]):.4f}")

    # Q2: Best alpha
    print("\nBest ARI by alpha (OSNet weight):")
    for alpha in alpha_values:
        sub = [r for r in results if r['alpha'] == alpha]
        aris = [r['ari'] for r in sub]
        print(f"  alpha={alpha:.1f}: mean ARI={np.mean(aris):.4f}  max ARI={np.max(aris):.4f}  "
              f"top5_mean={np.mean(sorted(aris, reverse=True)[:5]):.4f}")

    # Q3: Interaction between change_thresh and cluster_thresh
    print("\nInteraction: change_thresh vs cluster_thresh (mean ARI across alpha + pca):")
    print(f"  {'':>10}", end="")
    for ct in cluster_threshs:
        print(f"  clu={ct:.2f}", end="")
    print()
    for cht in change_threshs:
        print(f"  chg={cht:.2f} ", end="")
        for ct in cluster_threshs:
            sub = [r for r in results if r['change_thresh'] == cht and r['cluster_thresh'] == ct]
            mean_ari = np.mean([r['ari'] for r in sub])
            print(f"   {mean_ari:.4f}", end="")
        print()

    # Q4: ARI ceiling analysis
    best_ari = results[0]['ari']
    n_above_06 = sum(1 for r in results if r['ari'] > 0.6)
    n_above_05 = sum(1 for r in results if r['ari'] > 0.5)
    print(f"\nARI ceiling analysis:")
    print(f"  Best ARI found: {best_ari:.4f}")
    print(f"  Configs with ARI > 0.6: {n_above_06}/{len(results)}")
    print(f"  Configs with ARI > 0.5: {n_above_05}/{len(results)}")
    print(f"  ARI > 0.6 threshold {'ACHIEVABLE' if n_above_06 > 0 else 'NOT ACHIEVED'}")

    # ── Validate best config on Demo dataset ──────────────────────────────────

    print("\n" + "=" * 70)
    print("=== DEMO DATASET VALIDATION (5 players) ===")
    print("=" * 70)

    # Pick best config by ARI
    best = results[0]
    print(f"\nBest WA Open config: alpha={best['alpha']:.1f}, pca={best['n_pca']}, "
          f"change_thresh={best['change_thresh']:.2f}, cluster_thresh={best['cluster_thresh']:.2f}")
    print(f"WA Open metrics: ARI={best['ari']:.4f}, covered={best['covered']}/{best['n_true']}, "
          f"purity={best['purity']:.3f}, n_clusters={best['n_clusters']}")

    # Build Demo hybrid embedding with best params
    demo_white = demo_whitened_by_dim[best['n_pca']]
    demo_hybrid = np.concatenate(
        [best['alpha'] * demo_osnet_norm, (1 - best['alpha']) * demo_white], axis=1
    )
    demo_hybrid /= np.linalg.norm(demo_hybrid, axis=1, keepdims=True) + 1e-8

    demo_labels, demo_n_clusters = temporal_cluster(
        demo_fnames, demo_hybrid, best['change_thresh'], best['cluster_thresh']
    )
    demo_m = compute_metrics(demo_labels, demo_fnames, true_player_demo, N_TRUE_DEMO)
    print(f"\nDemo validation:")
    print(f"  n_clusters={demo_m['n_clusters']}, covered={demo_m['covered']}/{N_TRUE_DEMO}, "
          f"purity={demo_m['purity']:.3f}, ARI={demo_m['ari']:.4f}")

    # Also try top Pareto config on Demo
    if pareto and pareto[0] is not best:
        pareto_best = pareto[0]
        print(f"\nBest Pareto config: alpha={pareto_best['alpha']:.1f}, pca={pareto_best['n_pca']}, "
              f"change_thresh={pareto_best['change_thresh']:.2f}, cluster_thresh={pareto_best['cluster_thresh']:.2f}")
        print(f"WA Open: ARI={pareto_best['ari']:.4f}, n_clusters={pareto_best['n_clusters']}, "
              f"covered={pareto_best['covered']}/{N_TRUE_DEMO}")

        pw = demo_whitened_by_dim[pareto_best['n_pca']]
        ph = np.concatenate(
            [pareto_best['alpha'] * demo_osnet_norm, (1 - pareto_best['alpha']) * pw], axis=1
        )
        ph /= np.linalg.norm(ph, axis=1, keepdims=True) + 1e-8
        pl, pnc = temporal_cluster(demo_fnames, ph, pareto_best['change_thresh'], pareto_best['cluster_thresh'])
        pm = compute_metrics(pl, demo_fnames, true_player_demo, N_TRUE_DEMO)
        print(f"  Demo: n_clusters={pm['n_clusters']}, covered={pm['covered']}/{N_TRUE_DEMO}, "
              f"purity={pm['purity']:.3f}, ARI={pm['ari']:.4f}")

    # ── Final recommendation ───────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("=== FINAL RECOMMENDATION ===")
    print("=" * 70)

    # Choose the config with highest ARI among those with 20/20 coverage
    full_coverage = [r for r in results if r['covered'] == N_TRUE_WA]
    if full_coverage:
        rec = max(full_coverage, key=lambda r: r['ari'])
        print(f"\nBest config with full 20/20 coverage:")
    else:
        rec = results[0]
        print(f"\nNo config achieved 20/20 coverage; best ARI config:")

    print(f"  alpha        = {rec['alpha']:.1f}  (OSNet weight; DINOv2 weight = {1-rec['alpha']:.1f})")
    print(f"  n_pca        = {rec['n_pca']}  (PCA dimensions for DINOv2 whitening)")
    print(f"  change_thresh = {rec['change_thresh']:.2f}  (temporal changepoint sensitivity)")
    print(f"  cluster_thresh = {rec['cluster_thresh']:.2f}  (agglomerative clustering threshold)")
    print(f"\nExpected performance on WA Open (20 players, 696 photos):")
    print(f"  ARI       = {rec['ari']:.4f}")
    print(f"  Coverage  = {rec['covered']}/{rec['n_true']} players")
    print(f"  Purity    = {rec['purity']:.3f}")
    print(f"  Composite = {rec['composite']:.3f}")
    print(f"  n_clusters = {rec['n_clusters']}")

    # Comparison to baseline ARI=0.577
    baseline_ari = 0.577
    if rec['ari'] > baseline_ari:
        improvement = (rec['ari'] - baseline_ari) / baseline_ari * 100
        print(f"\n  Improvement over baseline ARI={baseline_ari}: +{rec['ari']-baseline_ari:.4f} ({improvement:.1f}%)")
    else:
        print(f"\n  vs baseline ARI={baseline_ari}: {rec['ari']-baseline_ari:+.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()

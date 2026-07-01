"""
_agent_baseline.py

Per-photo embedding baseline: sweeps embedders × clustering algorithms
against ground-truth player labels for the WA Open 202605 dataset.

Run from project root:
    python _agent_baseline.py

Outputs a ranked table of all clusterer × embedder combos by ARI,
plus a per-player analysis for the best config.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ── Project root on path ───────────────────────────────────────────────────────
ROOT = Path(r"C:\Users\erikg\OneDrive\Desktop\CodePlayground\badminton-photo-editor")
sys.path.insert(0, str(ROOT))

from inference.player_coverage import build_detector, build_embedder, embed_crops
from data.raw_reader import extract_thumbnail

# ── Dataset ────────────────────────────────────────────────────────────────────
RAW_DIR = Path(r"C:\Users\erikg\Downloads\Jay Ma Photography Training\202605 WA Open MS Raws")

RANGES = [
    (1,   78,  1),
    (79,  107, 2),
    (108, 126, 3),
    (127, 150, 2),
    (151, 177, 4),
    (178, 186, 5),
    (187, 194, 6),
    (195, 227, 5),
    (228, 252, 6),
    (253, 290, 7),
    (291, 296, 8),
    (297, 327, 9),
    (328, 347, 10),
    (348, 401, 11),
    (402, 442, 12),
    (443, 473, 13),
    (474, 485, 14),
    (486, 509, 15),
    (510, 527, 14),
    (528, 575, 16),
    (576, 595, 17),
    (596, 634, 18),
    (635, 653, 19),
    (654, 696, 20),
]

def num_to_player(n: int) -> int:
    for lo, hi, pid in RANGES:
        if lo <= n <= hi:
            return pid
    return -1


def collect_files():
    """Return sorted list of (filepath, file_number, player_id) for all 696 files."""
    files = sorted(RAW_DIR.glob("*.CR3"), key=lambda p: int(p.stem.replace("0V2A", "")))
    result = []
    for f in files:
        try:
            n = int(f.stem.replace("0V2A", ""))
        except ValueError:
            continue
        pid = num_to_player(n)
        if pid == -1:
            continue
        result.append((f, n, pid))
    return result


# ── Embedding ──────────────────────────────────────────────────────────────────

def build_embedder_by_name(name: str, device):
    import timm
    import torch
    model = timm.create_model(name, pretrained=True, num_classes=0, global_pool='token')
    return model.eval().to(device)


def embed_one(img, crop, embedder, device, embed_dim: int):
    """Embed a single crop; returns 1-D float32 array or None."""
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    _tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    try:
        t = _tf(crop).unsqueeze(0).to(device)
        # DINOv2 variants work best at 518×518; standard ViT at 224×224
        if '518' in str(embed_dim) or 'dinov2' in str(embedder.__class__):
            t = F.interpolate(t, size=(518, 518), mode='bilinear', align_corners=False)
        with torch.no_grad():
            feat = embedder(t)
            feat = F.normalize(feat, dim=1)
        return feat.cpu().float().numpy()[0]
    except Exception as e:
        return None


def _detect_largest_crop(img, detector):
    results = detector(img, classes=[0], verbose=False)
    w, h = img.size
    min_area = w * h * 0.01
    best_box, best_area = None, 0.0
    for r in results:
        for box in r.boxes.xyxy.cpu().tolist():
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area and area > best_area:
                best_box, best_area = (x1, y1, x2, y2), area
    return img.crop(best_box) if best_box else None


def extract_embeddings(files, detector, embedder, device, embed_dim, label=""):
    """Detect + embed each file; return (embeddings_array, gt_labels, file_indices_used)."""
    import torch.nn.functional as F
    import torch
    from torchvision import transforms

    _tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    embs, labels, paths_used = [], [], []
    n = len(files)
    t0 = time.time()

    for i, (fpath, fnum, pid) in enumerate(files):
        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {i}/{n}  ({elapsed:.0f}s elapsed)", flush=True)
        try:
            img = extract_thumbnail(fpath, size=(512, 512))
            crop = _detect_largest_crop(img, detector)
            if crop is None:
                continue
            t = _tf(crop).unsqueeze(0).to(device)
            # Use 518×518 for DINOv2 variants, 224 for others
            if 'dinov2' in label.lower():
                t = F.interpolate(t, size=(518, 518), mode='bilinear', align_corners=False)
            with torch.no_grad():
                feat = embedder(t)
                feat = F.normalize(feat, dim=1)
            e = feat.cpu().float().numpy()[0]
            embs.append(e)
            labels.append(pid)
            paths_used.append(fpath)
        except Exception as ex:
            pass

    print(f"  [{label}] Done: {len(embs)}/{n} photos embedded in {time.time()-t0:.0f}s", flush=True)
    return np.array(embs, dtype=np.float32), np.array(labels, dtype=int), paths_used


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(pred_labels: np.ndarray, gt_labels: np.ndarray, true_k: int = 20):
    """
    Returns dict with:
      n_clusters, coverage (fraction of true players seen), purity,
      ARI, NMI, composite (coverage * purity).
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from collections import Counter

    # Filter out noise (-1 from DBSCAN/HDBSCAN)
    mask = pred_labels != -1
    pred_f = pred_labels[mask]
    gt_f = gt_labels[mask]

    if len(pred_f) == 0:
        return dict(n_clusters=0, coverage=0.0, purity=0.0, ari=0.0, nmi=0.0, composite=0.0,
                    n_noise=int((pred_labels == -1).sum()))

    unique_clusters = np.unique(pred_f)
    n_clusters = len(unique_clusters)

    # Coverage: how many true players appear as the majority label in some cluster
    covered_players = set()
    pure_clusters = 0
    for c in unique_clusters:
        idx = np.where(pred_f == c)[0]
        gt_in_cluster = gt_f[idx]
        majority_player, majority_count = Counter(gt_in_cluster).most_common(1)[0]
        covered_players.add(majority_player)
        if majority_count == len(idx):
            pure_clusters += 1

    coverage = len(covered_players) / true_k
    purity = pure_clusters / n_clusters if n_clusters > 0 else 0.0

    ari = adjusted_rand_score(gt_f, pred_f)
    nmi = normalized_mutual_info_score(gt_f, pred_f, average_method='arithmetic')
    composite = coverage * purity

    n_noise = int((pred_labels == -1).sum())
    return dict(
        n_clusters=n_clusters,
        coverage=coverage,
        purity=purity,
        ari=ari,
        nmi=nmi,
        composite=composite,
        n_noise=n_noise,
    )


# ── Clustering algorithms ──────────────────────────────────────────────────────

def run_agglomerative(embs: np.ndarray, threshold: float):
    from sklearn.cluster import AgglomerativeClustering
    c = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='cosine',
        linkage='average',
    )
    return c.fit_predict(embs)


def run_dbscan(embs: np.ndarray, eps: float, min_samples: int):
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_distances
    D = cosine_distances(embs).astype(np.float64)
    c = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    return c.fit_predict(D)


def run_hdbscan(embs: np.ndarray, min_cluster_size: int):
    try:
        import hdbscan
        c = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
        return c.fit_predict(embs)
    except ImportError:
        return None


def run_kmeans(embs: np.ndarray, k: int):
    from sklearn.cluster import KMeans
    # L2-normalize before KMeans for cosine-like behaviour
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs_n = embs / norms
    c = KMeans(n_clusters=k, random_state=42, n_init=10)
    return c.fit_predict(embs_n)


def run_gmm(embs: np.ndarray, k: int):
    from sklearn.mixture import GaussianMixture
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs_n = embs / norms
    g = GaussianMixture(n_components=k, covariance_type='full', random_state=42, max_iter=200)
    return g.fit_predict(embs_n)


def sweep_clusterers(embs: np.ndarray, gt: np.ndarray, true_k: int = 20):
    """
    Run all clusterer variants. Returns list of (config_name, metrics_dict).
    """
    results = []
    n = len(embs)

    print("  Sweeping AgglomerativeClustering...", flush=True)
    for thr in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        try:
            pred = run_agglomerative(embs, thr)
            m = compute_metrics(pred, gt, true_k)
            results.append((f"Agglo(thr={thr:.2f})", m))
        except Exception as e:
            print(f"    Agglo thr={thr} failed: {e}")

    print("  Sweeping DBSCAN...", flush=True)
    for eps in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        for ms in [2, 3, 5]:
            try:
                pred = run_dbscan(embs, eps, ms)
                m = compute_metrics(pred, gt, true_k)
                results.append((f"DBSCAN(eps={eps},ms={ms})", m))
            except Exception as e:
                print(f"    DBSCAN eps={eps} ms={ms} failed: {e}")

    print("  Trying HDBSCAN...", flush=True)
    for mcs in [3, 5, 10]:
        pred = run_hdbscan(embs, mcs)
        if pred is None:
            print("    HDBSCAN not available (pip install hdbscan)")
            break
        m = compute_metrics(pred, gt, true_k)
        results.append((f"HDBSCAN(mcs={mcs})", m))

    print("  Sweeping KMeans...", flush=True)
    for k in [15, 20, 25]:
        try:
            pred = run_kmeans(embs, k)
            m = compute_metrics(pred, gt, true_k)
            results.append((f"KMeans(k={k})", m))
        except Exception as e:
            print(f"    KMeans k={k} failed: {e}")

    print("  Sweeping GMM...", flush=True)
    for k in [15, 20, 25]:
        try:
            pred = run_gmm(embs, k)
            m = compute_metrics(pred, gt, true_k)
            results.append((f"GMM(k={k})", m))
        except Exception as e:
            print(f"    GMM k={k} failed: {e}")

    return results


# ── Per-player analysis ────────────────────────────────────────────────────────

def per_player_analysis(embs: np.ndarray, gt: np.ndarray, pred: np.ndarray, true_k: int = 20):
    """Print intra/inter distance breakdown and confusion matrix."""
    from collections import Counter, defaultdict

    print("\n" + "="*70)
    print("PER-PLAYER ANALYSIS (best config)")
    print("="*70)

    # Intra-player distances
    print("\n--- Intra-player cosine distances (lower = tighter cluster) ---")
    player_embs = defaultdict(list)
    for e, pid in zip(embs, gt):
        player_embs[pid].append(e)

    intra_dists = {}
    for pid in sorted(player_embs):
        es = np.array(player_embs[pid])
        if len(es) < 2:
            intra_dists[pid] = 0.0
            continue
        # mean pairwise cosine distance
        from sklearn.metrics.pairwise import cosine_distances
        D = cosine_distances(es)
        # upper triangle
        triu = D[np.triu_indices(len(es), k=1)]
        intra_dists[pid] = float(triu.mean())
        print(f"  Player {pid:2d}: n={len(es):3d}  intra_dist={intra_dists[pid]:.4f}")

    mean_intra = np.mean(list(intra_dists.values()))

    # Inter-player distances (centroid-centroid)
    centroids = {}
    for pid, es in player_embs.items():
        c = np.mean(es, axis=0)
        c /= np.linalg.norm(c) + 1e-8
        centroids[pid] = c

    pids_sorted = sorted(centroids.keys())
    cent_matrix = np.stack([centroids[p] for p in pids_sorted])
    from sklearn.metrics.pairwise import cosine_distances
    inter_D = cosine_distances(cent_matrix)
    triu_inter = inter_D[np.triu_indices(len(pids_sorted), k=1)]
    mean_inter = float(triu_inter.mean())

    print(f"\n  Mean intra-player dist : {mean_intra:.4f}")
    print(f"  Mean inter-player dist : {mean_inter:.4f}")
    ratio = mean_inter / (mean_intra + 1e-8)
    print(f"  Separation ratio (inter/intra): {ratio:.2f}x")
    if ratio < 1.5:
        print("  *** LOW separation: players occupy the same embedding neighbourhood ***")
    elif ratio < 3.0:
        print("  *** MODERATE separation: some structure, but clusters overlap ***")
    else:
        print("  *** GOOD separation: inter >> intra ***")

    # Confusion: for each true player, which cluster(s) do their photos end up in?
    print("\n--- Per-player confusion (which pred clusters contain each player?) ---")
    for pid in sorted(player_embs.keys()):
        indices = np.where(gt == pid)[0]
        pred_cls = pred[indices]
        counter = Counter(pred_cls.tolist())
        top = counter.most_common(3)
        total = len(indices)
        frag = f"frag={len(counter)}" if len(counter) > 1 else "PURE"
        top_str = ", ".join(f"cl{c}:{cnt}" for c, cnt in top)
        print(f"  Player {pid:2d}: n={total:3d}  {frag:8s}  {top_str}")

    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"RAW dir: {RAW_DIR}")

    # Collect file list
    files = collect_files()
    print(f"Found {len(files)} files with ground-truth labels")
    unique_players = len(set(p for _, _, p in files))
    print(f"Unique players: {unique_players}")

    # ── Step 1: Build detector (shared across all embedders) ──────────────────
    print("\nLoading YOLO detector...")
    detector = build_detector()

    # ── Step 2: Embedder sweep ────────────────────────────────────────────────
    EMBEDDERS = [
        ("DINOv2-base",  "vit_base_patch14_reg4_dinov2"),
        ("DINOv2-small", "vit_small_patch14_reg4_dinov2"),
        ("DINOv2-large", "vit_large_patch14_reg4_dinov2"),
        ("ViT-B16-IN21k", "vit_base_patch16_224.augreg_in21k"),
    ]

    all_results = []   # list of (embedder_name, clusterer_name, metrics)
    emb_cache = {}     # cache embeddings per embedder so we don't re-compute

    for emb_name, emb_model_name in EMBEDDERS:
        print(f"\n{'='*60}")
        print(f"Embedder: {emb_name}  ({emb_model_name})")
        print("="*60)

        # Build embedder
        try:
            print(f"  Loading {emb_model_name}...", flush=True)
            embedder = build_embedder_by_name(emb_model_name, device)
            print(f"  Model loaded.", flush=True)
        except Exception as e:
            print(f"  FAILED to load {emb_model_name}: {e}")
            continue

        # Extract embeddings
        embs, gt, paths = extract_embeddings(files, detector, embedder, device, 768, label=emb_name)
        del embedder  # free VRAM

        if len(embs) < 10:
            print(f"  Too few embeddings ({len(embs)}), skipping.")
            continue

        emb_cache[emb_name] = (embs, gt, paths)
        print(f"  Embeddings: {embs.shape}, players: {len(set(gt.tolist()))}")

        # Sweep clusterers
        clusterer_results = sweep_clusterers(embs, gt, true_k=20)
        for cfg_name, metrics in clusterer_results:
            all_results.append((emb_name, cfg_name, metrics))

        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Step 3: Ranked table ──────────────────────────────────────────────────
    print("\n" + "="*100)
    print("RANKED RESULTS (by ARI, descending)")
    print("="*100)

    all_results_sorted = sorted(all_results, key=lambda x: x[2]['ari'], reverse=True)

    hdr = f"{'Rank':>4}  {'Embedder':<18}  {'Clusterer':<28}  {'K':>5}  {'Cov':>5}  {'Pur':>5}  {'ARI':>6}  {'NMI':>6}  {'Comp':>6}  {'Noise':>5}"
    print(hdr)
    print("-" * len(hdr))

    for rank, (emb_name, cfg_name, m) in enumerate(all_results_sorted, 1):
        print(
            f"{rank:4d}  {emb_name:<18}  {cfg_name:<28}  "
            f"{m['n_clusters']:5d}  {m['coverage']:5.2f}  {m['purity']:5.2f}  "
            f"{m['ari']:6.3f}  {m['nmi']:6.3f}  {m['composite']:6.3f}  "
            f"{m.get('n_noise',0):5d}"
        )

    # ── Step 4: Per-player analysis for best config ───────────────────────────
    if all_results_sorted:
        best_emb, best_cfg, best_m = all_results_sorted[0]
        print(f"\nBEST CONFIG: {best_emb} + {best_cfg}")
        print(f"  ARI={best_m['ari']:.4f}  NMI={best_m['nmi']:.4f}  "
              f"Coverage={best_m['coverage']:.2f}  Purity={best_m['purity']:.2f}")

        if best_emb in emb_cache:
            embs_best, gt_best, _ = emb_cache[best_emb]
            # Re-run the best clusterer to get labels
            if 'Agglo' in best_cfg:
                thr = float(best_cfg.split('=')[1].rstrip(')'))
                pred_best = run_agglomerative(embs_best, thr)
            elif 'DBSCAN' in best_cfg:
                parts = best_cfg.replace('DBSCAN(','').rstrip(')').split(',')
                eps = float(parts[0].split('=')[1])
                ms  = int(parts[1].split('=')[1])
                pred_best = run_dbscan(embs_best, eps, ms)
            elif 'HDBSCAN' in best_cfg:
                mcs = int(best_cfg.split('=')[1].rstrip(')'))
                pred_best = run_hdbscan(embs_best, mcs)
            elif 'KMeans' in best_cfg:
                k = int(best_cfg.split('=')[1].rstrip(')'))
                pred_best = run_kmeans(embs_best, k)
            elif 'GMM' in best_cfg:
                k = int(best_cfg.split('=')[1].rstrip(')'))
                pred_best = run_gmm(embs_best, k)
            else:
                pred_best = None

            if pred_best is not None:
                per_player_analysis(embs_best, gt_best, pred_best, true_k=20)

    # ── Summary statement ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    # Show best ARI per embedder
    from collections import defaultdict
    best_per_emb = defaultdict(lambda: -999)
    for emb, cfg, m in all_results:
        if m['ari'] > best_per_emb[emb]:
            best_per_emb[emb] = m['ari']
    for emb in [e for e, _ in EMBEDDERS]:
        if emb in best_per_emb:
            print(f"  {emb:<18}: best ARI = {best_per_emb[emb]:.4f}")

    # Separation verdict (from best embedder)
    if all_results_sorted:
        best_emb_name = all_results_sorted[0][0]
        if best_emb_name in emb_cache:
            embs_b, gt_b, _ = emb_cache[best_emb_name]
            from collections import defaultdict as dd
            from sklearn.metrics.pairwise import cosine_distances
            player_embs_b = dd(list)
            for e, pid in zip(embs_b, gt_b):
                player_embs_b[pid].append(e)
            intras = []
            for pid, es in player_embs_b.items():
                if len(es) >= 2:
                    D = cosine_distances(np.array(es))
                    intras.append(D[np.triu_indices(len(es), k=1)].mean())
            centroids_b = {}
            for pid, es in player_embs_b.items():
                c = np.mean(es, axis=0); c /= np.linalg.norm(c) + 1e-8
                centroids_b[pid] = c
            cm = np.stack(list(centroids_b.values()))
            iD = cosine_distances(cm)
            mean_inter = iD[np.triu_indices(len(cm), k=1)].mean()
            mean_intra = np.mean(intras)
            ratio = mean_inter / (mean_intra + 1e-8)
            print(f"\n  [{best_emb_name}] Inter/intra separation ratio: {ratio:.2f}x")
            if ratio < 2.0:
                print("  VERDICT: DINOv2 does NOT meaningfully separate the 20 badminton players.")
                print("  Players collapse into the same embedding neighbourhood.")
            elif ratio < 4.0:
                print("  VERDICT: DINOv2 has PARTIAL separation — some clusters are clean but others merge.")
            else:
                print("  VERDICT: DINOv2 DOES meaningfully separate players.")


if __name__ == "__main__":
    main()

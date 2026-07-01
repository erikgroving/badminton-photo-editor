"""
Color + Appearance Fusion agent for player identification.

Tests three approaches:
  A: HSV color histogram (PCA-reduced) + DINOv2 embedding fusion
  B: Dominant color KMeans (9-dim) clustering
  C: Color-histogram temporal changepoints merged with embedding changepoints

Dataset: 202605 WA Open MS Raws, files 0001-0696, 20 true players.

Run from project root:
  cd "C:\\Users\\erikg\\OneDrive\\Desktop\\CodePlayground\\badminton-photo-editor"
  python _agent_color.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r'C:\Users\erikg\OneDrive\Desktop\CodePlayground\badminton-photo-editor')

from inference.player_coverage import build_detector, build_embedder, embed_crops, cluster_embeddings
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
N_TRUE_PLAYERS = 20

def file_num(fname: str) -> int:
    m = re.search(r'(\d{4,})', Path(fname).stem)
    return int(m.group(1)) if m else -1

def true_player(fname: str) -> int:
    n = file_num(fname)
    for s, e, pid in RANGES:
        if s <= n <= e:
            return pid
    return 0

# ── Color feature extraction ────────────────────────────────────────────────────

def hsv_histogram(crop_rgb, n_bins: int = 32) -> np.ndarray:
    """
    Compute a normalized HSV histogram of the upper-body sub-crop.
    Returns a flat array of shape (3 * n_bins,).
    """
    from PIL import Image
    import cv2

    # Take top 60% = torso / jersey region
    w, h = crop_rgb.size
    upper = crop_rgb.crop((0, 0, w, int(h * 0.60)))

    arr = np.array(upper, dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    # Per-channel histograms, normalized
    hists = []
    ranges_ch = [(0, 180), (0, 256), (0, 256)]
    for ch in range(3):
        h_vals = cv2.calcHist([hsv], [ch], None, [n_bins], [ranges_ch[ch][0], ranges_ch[ch][1]])
        h_vals = h_vals.flatten().astype(np.float32)
        total = h_vals.sum()
        if total > 0:
            h_vals /= total
        hists.append(h_vals)

    return np.concatenate(hists)  # shape (3*n_bins,)


def dominant_colors_hsv(crop_rgb, k: int = 3) -> np.ndarray:
    """
    Extract K dominant colors using KMeans on upper-body pixels.
    Returns a flat (k*3,) HSV vector sorted by cluster size (dominant first).
    """
    from PIL import Image
    from sklearn.cluster import MiniBatchKMeans
    import cv2

    w, h = crop_rgb.size
    upper = crop_rgb.crop((0, 0, w, int(h * 0.60)))
    arr = np.array(upper, dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    pixels = hsv.reshape(-1, 3).astype(np.float32)
    # Subsample for speed
    if len(pixels) > 2000:
        idx = np.random.choice(len(pixels), 2000, replace=False)
        pixels = pixels[idx]

    if len(pixels) < k:
        # pad with zeros
        return np.zeros(k * 3, dtype=np.float32)

    km = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42)
    labels = km.fit_predict(pixels)
    centers = km.cluster_centers_  # shape (k, 3)

    # Sort by cluster size descending so dominant color is first
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    sorted_centers = centers[order]

    # Normalize each channel to [0,1]
    norm = sorted_centers.copy()
    norm[:, 0] /= 180.0   # Hue: 0-180
    norm[:, 1] /= 255.0   # Sat: 0-255
    norm[:, 2] /= 255.0   # Val: 0-255

    return norm.flatten().astype(np.float32)


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def evaluate(labels: np.ndarray, names: list[str], n_true: int = N_TRUE_PLAYERS) -> dict:
    """Compute coverage, purity, and ARI."""
    from sklearn.metrics import adjusted_rand_score

    player_to_clusters: dict[int, set] = {}
    cluster_to_players: dict[int, set] = {}
    gt_labels = []
    for i, fname in enumerate(names):
        pid = true_player(fname)
        cid = int(labels[i])
        player_to_clusters.setdefault(pid, set()).add(cid)
        cluster_to_players.setdefault(cid, set()).add(pid)
        gt_labels.append(pid)

    n_clusters = int(labels.max()) + 1
    pure = sum(1 for ps in cluster_to_players.values() if len(ps) == 1)
    covered = set(player_to_clusters.keys()) - {0}
    coverage_frac = len(covered) / n_true

    purity = pure / n_clusters if n_clusters > 0 else 0.0
    composite = coverage_frac * purity

    # ARI (only on labeled photos, pid > 0)
    gt_arr = np.array(gt_labels)
    pred_arr = labels
    mask = gt_arr > 0
    if mask.sum() > 1:
        ari = adjusted_rand_score(gt_arr[mask], pred_arr[mask])
    else:
        ari = 0.0

    return {
        'n_clusters': n_clusters,
        'n_pure': pure,
        'covered': len(covered),
        'coverage_frac': coverage_frac,
        'purity': purity,
        'composite': composite,
        'ari': ari,
    }


def print_eval(tag: str, metrics: dict):
    print(f"  [{tag}]  clusters={metrics['n_clusters']:3d}  "
          f"covered={metrics['covered']}/{N_TRUE_PLAYERS}  "
          f"purity={metrics['purity']:.3f}  "
          f"ARI={metrics['ari']:.3f}  "
          f"composite={metrics['composite']:.3f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    raw_dir = Path(r'C:\Users\erikg\Downloads\Jay Ma Photography Training\202605 WA Open MS Raws')

    # Gather labeled files
    _seen: set[str] = set()
    labeled_raws = []
    for p in sorted(raw_dir.glob('*.CR3'), key=lambda x: file_num(x.name)):
        if p.name.lower() in _seen:
            continue
        _seen.add(p.name.lower())
        n = file_num(p.name)
        if 1 <= n <= 696:
            labeled_raws.append(p)

    # Also check .cr3 lowercase just in case
    for p in sorted(raw_dir.glob('*.cr3'), key=lambda x: file_num(x.name)):
        if p.name.lower() in _seen:
            continue
        _seen.add(p.name.lower())
        n = file_num(p.name)
        if 1 <= n <= 696:
            labeled_raws.append(p)
    labeled_raws = sorted(labeled_raws, key=lambda x: file_num(x.name))

    print(f"Labeled photos: {len(labeled_raws)}")
    print(f"True players:   {N_TRUE_PLAYERS}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Build models
    print("Loading YOLO detector...")
    detector = build_detector()
    print("Loading DINOv2 embedder...")
    embedder = build_embedder(device)

    # ── Phase 1: Detect, embed, and extract color features ────────────────────
    print("\nDetecting + extracting features (this will take a few minutes)...")
    t0 = time.time()

    photo_names: list[str] = []
    dino_embs: list[np.ndarray] = []
    color_hists: list[np.ndarray] = []   # HSV histograms
    dom_colors: list[np.ndarray] = []    # Dominant color vectors

    for i, p in enumerate(labeled_raws):
        img = extract_thumbnail(p, size=(448, 448))
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

        # Largest box
        largest = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        crop = img.crop(largest)

        # DINOv2 embedding
        try:
            emb = embed_crops([crop], embedder, device)[0]
        except Exception as ex:
            print(f"  embed error {p.name}: {ex}")
            continue

        # Color features
        try:
            hist = hsv_histogram(crop, n_bins=32)
            dom = dominant_colors_hsv(crop, k=3)
        except Exception as ex:
            print(f"  color error {p.name}: {ex}")
            continue

        photo_names.append(p.name)
        dino_embs.append(emb)
        color_hists.append(hist)
        dom_colors.append(dom)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(labeled_raws)} done ({len(photo_names)} detected) "
                  f"[{elapsed:.0f}s]")

    elapsed = time.time() - t0
    print(f"\nPhotos with detections: {len(photo_names)}/{len(labeled_raws)}  [{elapsed:.0f}s total]\n")

    dino_matrix = np.stack(dino_embs)          # (N, 768)
    hist_matrix = np.stack(color_hists)        # (N, 96) for 3*32
    dom_matrix  = np.stack(dom_colors)         # (N, 9)

    # ── Baseline: DINOv2 only ─────────────────────────────────────────────────
    print("=" * 70)
    print("BASELINE: DINOv2 only")
    print("=" * 70)
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        labels = cluster_embeddings(dino_matrix, thresh)
        m = evaluate(labels, photo_names)
        print_eval(f"DINOv2 t={thresh:.2f}", m)

    # ── Approach B: Dominant Colors Only ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("APPROACH B: Dominant Colors Only (9-dim KMeans centroids)")
    print("=" * 70)

    # Normalize dominant color matrix
    dom_norm = dom_matrix.copy()
    norms = np.linalg.norm(dom_norm, axis=1, keepdims=True)
    dom_norm = dom_norm / (norms + 1e-8)

    for thresh in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        labels = cluster_embeddings(dom_norm, thresh)
        m = evaluate(labels, photo_names)
        print_eval(f"DomColor t={thresh:.2f}", m)

    # ── Approach A: Color Histogram + DINOv2 Fusion ───────────────────────────
    print("\n" + "=" * 70)
    print("APPROACH A: HSV Histogram (PCA-reduced) + DINOv2 Fusion")
    print("=" * 70)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # PCA reduce color histogram: 96 → 20 dims
    pca_n = min(20, hist_matrix.shape[0] - 1)
    scaler = StandardScaler()
    hist_scaled = scaler.fit_transform(hist_matrix)
    pca = PCA(n_components=pca_n, random_state=42)
    hist_pca = pca.fit_transform(hist_scaled)   # (N, pca_n)
    print(f"  Histogram PCA: {hist_matrix.shape[1]}d → {pca_n}d "
          f"(explained var: {pca.explained_variance_ratio_.sum():.3f})")

    # L2-normalize the PCA-reduced histogram
    hist_norms = np.linalg.norm(hist_pca, axis=1, keepdims=True)
    hist_pca_norm = hist_pca / (hist_norms + 1e-8)

    best_A = None
    best_A_score = -1.0
    best_A_tag = ""

    for alpha in [0.3, 0.5, 0.7, 0.9]:
        # alpha = weight on DINOv2; (1-alpha) = weight on color histogram
        fused = np.concatenate([
            alpha * dino_matrix,
            (1 - alpha) * hist_pca_norm,
        ], axis=1)
        # L2-normalize the fused vector
        fnorms = np.linalg.norm(fused, axis=1, keepdims=True)
        fused_norm = fused / (fnorms + 1e-8)

        for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
            labels = cluster_embeddings(fused_norm, thresh)
            m = evaluate(labels, photo_names)
            tag = f"A alpha={alpha:.1f} t={thresh:.2f}"
            print_eval(tag, m)
            if m['composite'] > best_A_score:
                best_A_score = m['composite']
                best_A = (labels, m)
                best_A_tag = tag
        print()

    # ── Approach C: Color Temporal Fusion ────────────────────────────────────
    print("\n" + "=" * 70)
    print("APPROACH C: Color Temporal Changepoint + DINOv2 Changepoint Union")
    print("=" * 70)

    def temporal_cluster_dual(names, dino_embs_list, hist_list,
                              color_change_thresh, dino_change_thresh, cluster_thresh,
                              color_weight=0.5):
        """
        Build segments from UNION of:
          - color histogram changepoints (consecutive chi-squared distance > color_change_thresh)
          - DINOv2 embedding changepoints (consecutive cosine distance > dino_change_thresh)
        Then average fused embeddings within each segment and cluster.
        """
        order = sorted(range(len(names)), key=lambda i: file_num(names[i]))
        sorted_names = [names[i] for i in order]
        sorted_dino  = [dino_embs_list[i] for i in order]
        sorted_hist  = [hist_list[i] for i in order]

        def chi2_dist(a, b):
            """Chi-squared distance between two normalized histograms."""
            denom = a + b + 1e-8
            return float(np.sum((a - b) ** 2 / denom))

        def cos_dist(a, b):
            an = a / (np.linalg.norm(a) + 1e-8)
            bn = b / (np.linalg.norm(b) + 1e-8)
            return 1.0 - float(np.dot(an, bn))

        # Build segments: new segment when EITHER changepoint fires
        segments = []  # list of (names, avg_dino, avg_hist)
        seg_names = [sorted_names[0]]
        seg_dino  = [sorted_dino[0]]
        seg_hist  = [sorted_hist[0]]
        seg_dino_avg = sorted_dino[0] / (np.linalg.norm(sorted_dino[0]) + 1e-8)
        seg_hist_avg = sorted_hist[0].copy()

        for i in range(1, len(sorted_names)):
            d_color = chi2_dist(sorted_hist[i], seg_hist_avg)
            d_dino  = cos_dist(sorted_dino[i], seg_dino_avg)

            new_segment = (d_color > color_change_thresh) or (d_dino > dino_change_thresh)

            if new_segment:
                avg_d = np.mean(seg_dino, axis=0); avg_d /= np.linalg.norm(avg_d) + 1e-8
                avg_h = np.mean(seg_hist, axis=0)
                segments.append((seg_names, avg_d, avg_h))
                seg_names = [sorted_names[i]]
                seg_dino  = [sorted_dino[i]]
                seg_hist  = [sorted_hist[i]]
                seg_dino_avg = sorted_dino[i] / (np.linalg.norm(sorted_dino[i]) + 1e-8)
                seg_hist_avg = sorted_hist[i].copy()
            else:
                seg_names.append(sorted_names[i])
                seg_dino.append(sorted_dino[i])
                seg_hist.append(sorted_hist[i])
                seg_dino_avg = np.mean(seg_dino, axis=0)
                seg_dino_avg /= np.linalg.norm(seg_dino_avg) + 1e-8
                seg_hist_avg = np.mean(seg_hist, axis=0)

        avg_d = np.mean(seg_dino, axis=0); avg_d /= np.linalg.norm(avg_d) + 1e-8
        avg_h = np.mean(seg_hist, axis=0)
        segments.append((seg_names, avg_d, avg_h))

        # Build fused segment representations
        seg_reprs = []
        for _, avg_d, avg_h in segments:
            # PCA-reduce color and fuse
            h_scaled = scaler.transform(avg_h.reshape(1, -1))
            h_pca = pca.transform(h_scaled)[0]
            h_norm = h_pca / (np.linalg.norm(h_pca) + 1e-8)
            fused = np.concatenate([
                (1 - color_weight) * avg_d,
                color_weight * h_norm,
            ])
            fused /= np.linalg.norm(fused) + 1e-8
            seg_reprs.append(fused)

        seg_matrix = np.stack(seg_reprs)
        if len(segments) == 1:
            seg_labels = np.array([0])
        else:
            seg_labels = cluster_embeddings(seg_matrix, cluster_thresh)

        # Map back to photo labels
        photo_label_map = {}
        for seg_idx, (seg_n, _, _) in enumerate(segments):
            cid = int(seg_labels[seg_idx])
            for fname in seg_n:
                photo_label_map[fname] = cid

        labels = np.array([photo_label_map.get(n, 0) for n in names])
        return labels, len(segments)

    best_C = None
    best_C_score = -1.0
    best_C_tag = ""

    for color_ct in [0.05, 0.10, 0.20]:
        for dino_ct in [0.15, 0.20, 0.25]:
            for cw in [0.3, 0.5]:
                for ht in [0.55, 0.60, 0.65]:
                    labels, n_segs = temporal_cluster_dual(
                        photo_names, dino_embs, color_hists,
                        color_ct, dino_ct, ht, cw
                    )
                    m = evaluate(labels, photo_names)
                    tag = (f"C cct={color_ct:.2f} dct={dino_ct:.2f} "
                           f"cw={cw:.1f} ht={ht:.2f} segs={n_segs}")
                    print_eval(tag, m)
                    if m['composite'] > best_C_score:
                        best_C_score = m['composite']
                        best_C = (labels, m)
                        best_C_tag = tag
        print()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Best Configuration per Approach")
    print("=" * 70)

    # Best baseline
    best_baseline_score = -1.0
    best_baseline = None
    best_baseline_tag = ""
    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
        labels = cluster_embeddings(dino_matrix, thresh)
        m = evaluate(labels, photo_names)
        if m['composite'] > best_baseline_score:
            best_baseline_score = m['composite']
            best_baseline = (labels, m)
            best_baseline_tag = f"DINOv2 t={thresh:.2f}"

    print(f"\nBaseline:    {best_baseline_tag}")
    print_eval(best_baseline_tag, best_baseline[1])

    if best_A is not None:
        print(f"\nApproach A:  {best_A_tag}")
        print_eval(best_A_tag, best_A[1])

    if best_C is not None:
        print(f"\nApproach C:  {best_C_tag}")
        print_eval(best_C_tag, best_C[1])

    # ── Visual sanity check on best overall ────────────────────────────────────
    # Pick the best composite score overall
    candidates = []
    if best_baseline is not None:
        candidates.append((best_baseline_score, best_baseline[0], best_baseline_tag))
    if best_A is not None:
        candidates.append((best_A_score, best_A[0], best_A_tag))
    if best_C is not None:
        candidates.append((best_C_score, best_C[0], best_C_tag))

    if candidates:
        _, best_labels, best_tag = max(candidates, key=lambda x: x[0])
        print(f"\n{'='*70}")
        print(f"VISUAL SANITY CHECK: Best config = [{best_tag}]")
        print(f"{'='*70}")

        # Per-cluster breakdown showing which true players land together
        cluster_to_players: dict[int, dict[int, int]] = {}  # cluster -> {player -> count}
        for i, fname in enumerate(photo_names):
            pid = true_player(fname)
            if pid == 0:
                continue
            cid = int(best_labels[i])
            cluster_to_players.setdefault(cid, {}).setdefault(pid, 0)
            cluster_to_players[cid][pid] += 1

        # Sort clusters by size
        cluster_sizes = {cid: sum(v.values()) for cid, v in cluster_to_players.items()}
        sorted_clusters = sorted(cluster_to_players.keys(),
                                 key=lambda c: cluster_sizes[c], reverse=True)

        print(f"\n{'Cluster':>8}  {'Size':>6}  Players (pid: count)")
        print("-" * 60)
        n_pure = 0
        n_mixed = 0
        for cid in sorted_clusters:
            players = cluster_to_players[cid]
            size = cluster_sizes[cid]
            n_pids = len(players)
            purity_marker = " [PURE]" if n_pids == 1 else f" [MIXED x{n_pids}]"
            players_str = "  ".join(f"P{pid}:{cnt}" for pid, cnt in
                                    sorted(players.items(), key=lambda x: -x[1]))
            print(f"  C{cid:04d}:  {size:5d}  {players_str}{purity_marker}")
            if n_pids == 1:
                n_pure += 1
            else:
                n_mixed += 1

        print(f"\nPure clusters: {n_pure}  Mixed clusters: {n_mixed}")

        # Check for same-team over-separation: are same players split across clusters?
        player_to_clusters: dict[int, set] = {}
        for i, fname in enumerate(photo_names):
            pid = true_player(fname)
            if pid > 0:
                cid = int(best_labels[i])
                player_to_clusters.setdefault(pid, set()).add(cid)

        print("\nPlayers split across multiple clusters (over-segmentation):")
        any_split = False
        for pid in sorted(player_to_clusters.keys()):
            clusters = player_to_clusters[pid]
            if len(clusters) > 1:
                print(f"  P{pid}: clusters {sorted(clusters)}")
                any_split = True
        if not any_split:
            print("  None — no player is split!")

        print("\nUndetected true players:")
        all_covered = set(player_to_clusters.keys())
        missing = set(range(1, N_TRUE_PLAYERS + 1)) - all_covered
        if missing:
            for pid in sorted(missing):
                print(f"  P{pid}: not detected in any photo")
        else:
            print("  All 20 players detected!")

    print("\n[Done]")


if __name__ == '__main__':
    main()

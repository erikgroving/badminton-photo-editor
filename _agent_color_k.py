"""
Forced-K clustering on fused HSV color + DINOv2 features.

Tests:
  1. Forced-K AgglomerativeClustering (cosine, average linkage)
  2. Forced-K KMeans
  3. Alpha sweep at K=20 to find optimal color weighting
  4. Fused features on sequential K-NN graph (ConnComp)

Dataset: 202605 WA Open MS Raws (0001-0696, 20 true players)
Reuses cached embeddings from prior agents where available.

Run from project root:
  python _agent_color_k.py
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

from inference.player_coverage import build_detector, build_embedder, embed_crops
from data.raw_reader import extract_thumbnail

# ── Paths ──────────────────────────────────────────────────────────────────────

RAW_DIR     = Path(r'C:\Users\erikg\Downloads\Jay Ma Photography Training\202605 WA Open MS Raws')
CACHE_DIR   = Path(r'C:\Users\erikg\OneDrive\Desktop\CodePlayground\badminton-photo-editor\cache')
SEQ_CACHE   = CACHE_DIR / '_agent_seq_embs.npz'
COLOR_CACHE = CACHE_DIR / '_agent_color_embs.npz'

# ── Ground truth ───────────────────────────────────────────────────────────────

RANGES = [
    (1,    78,   1), (79,   107,  2), (108,  126,  3), (127,  150,  2),
    (151,  177,  4), (178,  186,  5), (187,  194,  6), (195,  227,  5),
    (228,  252,  6), (253,  290,  7), (291,  296,  8), (297,  327,  9),
    (328,  347, 10), (348,  401, 11), (402,  442, 12), (443,  473, 13),
    (474,  485, 14), (486,  509, 15), (510,  527, 14), (528,  575, 16),
    (576,  595, 17), (596,  634, 18), (635,  653, 19), (654,  696, 20),
]
N_TRUE_PLAYERS = 20
GAP_PLAYERS = {2, 5, 6, 14}


def file_num(fname: str) -> int:
    m = re.search(r'(\d{4,})', Path(fname).stem)
    return int(m.group(1)) if m else -1


def true_player(fname: str) -> int:
    n = file_num(fname)
    for s, e, pid in RANGES:
        if s <= n <= e:
            return pid
    return 0


# ── Color feature extraction ───────────────────────────────────────────────────

def hsv_histogram(crop_rgb, n_bins: int = 32) -> np.ndarray:
    """Normalized HSV histogram of upper-body (top 60%). Returns (3*n_bins,)."""
    import cv2
    w, h = crop_rgb.size
    upper = crop_rgb.crop((0, 0, w, int(h * 0.60)))
    arr = np.array(upper, dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hists = []
    ranges_ch = [(0, 180), (0, 256), (0, 256)]
    for ch in range(3):
        h_vals = cv2.calcHist([hsv], [ch], None, [n_bins],
                              [ranges_ch[ch][0], ranges_ch[ch][1]])
        h_vals = h_vals.flatten().astype(np.float32)
        total = h_vals.sum()
        if total > 0:
            h_vals /= total
        hists.append(h_vals)
    return np.concatenate(hists)   # (96,)


# ── Load / build features ──────────────────────────────────────────────────────

def load_or_build_features() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns (dino_matrix [N,768], hist_matrix [N,96], names [N]).
    Tries to load from COLOR_CACHE first; falls back to full extraction.
    The sequential cache only has DINOv2 embeddings (no color), so we always
    need to check for the color cache separately.
    """
    if COLOR_CACHE.exists():
        print(f'Loading cached color+dino features from {COLOR_CACHE}')
        data = np.load(COLOR_CACHE, allow_pickle=True)
        dino  = data['dino_embs']
        hists = data['color_hists']
        names = list(data['names'])
        print(f'  Loaded {len(names)} photos with detections.')
        return dino, hists, names

    # Need to extract from scratch
    print('No color cache found — extracting features from RAW files...')

    _seen: set[str] = set()
    labeled_raws: list[Path] = []
    for p in sorted(RAW_DIR.glob('*'), key=lambda x: file_num(x.name)):
        if p.suffix.lower() != '.cr3' or p.name.lower() in _seen:
            continue
        _seen.add(p.name.lower())
        if 1 <= file_num(p.name) <= 696:
            labeled_raws.append(p)
    labeled_raws = sorted(labeled_raws, key=lambda x: file_num(x.name))
    print(f'  Found {len(labeled_raws)} labeled CR3 files.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')
    detector = build_detector()
    embedder = build_embedder(device)

    photo_names: list[str] = []
    dino_embs_list: list[np.ndarray] = []
    color_hists_list: list[np.ndarray] = []

    t0 = time.time()
    for i, p in enumerate(labeled_raws):
        try:
            img = extract_thumbnail(p, size=(448, 448))
        except Exception as ex:
            print(f'  thumb error {p.name}: {ex}')
            continue

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
        crop = img.crop(largest)

        try:
            emb = embed_crops([crop], embedder, device)[0]
        except Exception as ex:
            print(f'  embed error {p.name}: {ex}')
            continue

        try:
            hist = hsv_histogram(crop, n_bins=32)
        except Exception as ex:
            print(f'  color error {p.name}: {ex}')
            continue

        photo_names.append(p.name)
        dino_embs_list.append(emb)
        color_hists_list.append(hist)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f'  {i+1}/{len(labeled_raws)} done ({len(photo_names)} detected) [{elapsed:.0f}s]')

    elapsed = time.time() - t0
    print(f'  Extraction done: {len(photo_names)} detections / {len(labeled_raws)} files [{elapsed:.0f}s]')

    dino_matrix  = np.stack(dino_embs_list)
    hist_matrix  = np.stack(color_hists_list)
    names_arr    = np.array(photo_names)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(COLOR_CACHE, dino_embs=dino_matrix, color_hists=hist_matrix, names=names_arr)
    print(f'  Saved to {COLOR_CACHE}')
    return dino_matrix, hist_matrix, photo_names


# ── Fused feature builder ──────────────────────────────────────────────────────

def build_fused(dino_matrix: np.ndarray, hist_matrix: np.ndarray,
                alpha: float, pca_n: int = 20) -> np.ndarray:
    """
    alpha = weight on DINOv2; (1-alpha) = weight on HSV histogram.
    Returns L2-normalized fused matrix of shape (N, 768+pca_n).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    pca_n = min(pca_n, hist_matrix.shape[0] - 1, hist_matrix.shape[1])
    scaler = StandardScaler()
    hist_scaled = scaler.fit_transform(hist_matrix)
    pca = PCA(n_components=pca_n, random_state=42)
    hist_pca = pca.fit_transform(hist_scaled)

    # L2-normalize color component
    hn = np.linalg.norm(hist_pca, axis=1, keepdims=True)
    hist_norm = hist_pca / (hn + 1e-8)

    # DINOv2 is already L2-normalized from embed_crops()
    fused = np.concatenate([alpha * dino_matrix, (1 - alpha) * hist_norm], axis=1)
    fn = np.linalg.norm(fused, axis=1, keepdims=True)
    return fused / (fn + 1e-8)


# ── Metrics ────────────────────────────────────────────────────────────────────

def evaluate(labels: np.ndarray, names: list[str]) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    player_to_clusters: dict[int, set] = {}
    cluster_to_players: dict[int, set] = {}
    gt_labels = []

    for i, fname in enumerate(names):
        pid = true_player(fname)
        cid = int(labels[i])
        player_to_clusters.setdefault(pid, set()).add(cid)
        cluster_to_players.setdefault(cid, set()).add(pid)
        gt_labels.append(pid)

    n_clusters  = len(cluster_to_players)
    pure        = sum(1 for ps in cluster_to_players.values() if len(ps) == 1)
    covered     = set(player_to_clusters.keys()) - {0}
    coverage_f  = len(covered) / N_TRUE_PLAYERS
    purity      = pure / n_clusters if n_clusters > 0 else 0.0

    gt_arr   = np.array(gt_labels)
    pred_arr = labels
    mask     = gt_arr > 0

    if mask.sum() > 1:
        ari = adjusted_rand_score(gt_arr[mask], pred_arr[mask])
        nmi = normalized_mutual_info_score(gt_arr[mask], pred_arr[mask])
    else:
        ari = nmi = 0.0

    gap_merged = sum(
        1 for pid in GAP_PLAYERS
        if pid in player_to_clusters and len(player_to_clusters[pid]) == 1
    )

    return {
        'n_clusters': n_clusters,
        'n_pure':     pure,
        'covered':    len(covered),
        'coverage':   coverage_f,
        'purity':     purity,
        'ari':        ari,
        'nmi':        nmi,
        'composite':  coverage_f * purity,
        'gap_merged': gap_merged,
    }


def hdr(title: str):
    print(f'\n{"="*80}')
    print(title)
    print('='*80)
    print(f'  {"Config":<48} {"K":>4} {"cov":>6} {"pure/tot":>10} {"purity":>7} '
          f'{"ARI":>7} {"NMI":>7} {"gap":>5}')


def row(tag: str, m: dict):
    print(f'  {tag:<48} {m["n_clusters"]:4d} '
          f'{m["covered"]:2d}/{N_TRUE_PLAYERS}  '
          f'{m["n_pure"]:3d}/{m["n_clusters"]:3d}  '
          f'{m["purity"]:.3f}  {m["ari"]:.3f}  {m["nmi"]:.3f}  '
          f'{m["gap_merged"]}/{len(GAP_PLAYERS)}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('Loading / building features...')
    dino_matrix, hist_matrix, photo_names = load_or_build_features()
    N = len(photo_names)
    print(f'  N={N}  DINOv2={dino_matrix.shape}  HSV hist={hist_matrix.shape}')

    # ── Section 1: Forced-K AgglomerativeClustering ──────────────────────────
    hdr('SECTION 1: Forced-K AgglomerativeClustering (cosine, average) — alpha=0.3')

    from sklearn.cluster import AgglomerativeClustering

    alpha_default = 0.3
    fused_default = build_fused(dino_matrix, hist_matrix, alpha=alpha_default)

    best_agg = None
    for K in [15, 18, 20, 22, 25, 30]:
        model = AgglomerativeClustering(n_clusters=K, metric='cosine', linkage='average')
        labels = model.fit_predict(fused_default)
        m = evaluate(labels, photo_names)
        tag = f'Agg K={K:2d} alpha={alpha_default}'
        row(tag, m)
        if best_agg is None or m['ari'] > best_agg[0]:
            best_agg = (m['ari'], m, tag, labels)

    # ── Section 2: Forced-K KMeans ───────────────────────────────────────────
    hdr('SECTION 2: Forced-K KMeans — alpha=0.3')

    from sklearn.cluster import KMeans

    best_km = None
    for K in [15, 18, 20, 22, 25, 30]:
        labels = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(fused_default)
        m = evaluate(labels, photo_names)
        tag = f'KMeans K={K:2d} alpha={alpha_default}'
        row(tag, m)
        if best_km is None or m['ari'] > best_km[0]:
            best_km = (m['ari'], m, tag, labels)

    # ── Section 3: Alpha sweep at K=20 ──────────────────────────────────────
    hdr('SECTION 3: Alpha sweep (K=20 fixed) — AgglomerativeClustering + KMeans')
    print(f'  alpha=1.0 means pure DINOv2; alpha=0.0 means pure color')

    K_fixed = 20
    best_alpha_agg = None
    best_alpha_km  = None

    print(f'\n  -- AgglomerativeClustering (cosine, average) --')
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        fused = build_fused(dino_matrix, hist_matrix, alpha=alpha)
        model = AgglomerativeClustering(n_clusters=K_fixed, metric='cosine', linkage='average')
        labels = model.fit_predict(fused)
        m = evaluate(labels, photo_names)
        tag = f'Agg alpha={alpha:.1f} K={K_fixed}'
        row(tag, m)
        if best_alpha_agg is None or m['ari'] > best_alpha_agg[0]:
            best_alpha_agg = (m['ari'], m, tag, alpha)

    print(f'\n  -- KMeans --')
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        fused = build_fused(dino_matrix, hist_matrix, alpha=alpha)
        labels = KMeans(n_clusters=K_fixed, random_state=42, n_init=10).fit_predict(fused)
        m = evaluate(labels, photo_names)
        tag = f'KMeans alpha={alpha:.1f} K={K_fixed}'
        row(tag, m)
        if best_alpha_km is None or m['ari'] > best_alpha_km[0]:
            best_alpha_km = (m['ari'], m, tag, alpha)

    # ── Section 4: Sequential K-NN graph (ConnComp) ───────────────────────────
    hdr('SECTION 4: Sequential K-NN ConnComp graph — DINOv2-only vs Fused')

    import networkx as nx

    def knn_conncomp(embs_mat: np.ndarray, names: list[str], K: int, sim_thresh: float) -> np.ndarray:
        """
        Sequential K-NN connected-component clustering.
        Each photo i is connected to its K nearest sequential neighbours
        (i-K..i-1, i+1..i+K) if cosine sim >= sim_thresh.
        """
        n_norms = np.linalg.norm(embs_mat, axis=1, keepdims=True) + 1e-8
        e = embs_mat / n_norms
        N_loc = len(e)
        G = nx.Graph()
        G.add_nodes_from(range(N_loc))
        for i in range(N_loc):
            lo = max(0, i - K)
            hi = min(N_loc, i + K + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                sim = float(np.dot(e[i], e[j]))
                if sim >= sim_thresh:
                    if G.has_edge(i, j):
                        if G[i][j]['weight'] < sim:
                            G[i][j]['weight'] = sim
                    else:
                        G.add_edge(i, j, weight=sim)
        labels_out = np.zeros(N_loc, dtype=int)
        cid = 0
        for comp in nx.connected_components(G):
            for node in comp:
                labels_out[node] = cid
            cid += 1
        return labels_out

    # Sort photo_names by file number (sequential order)
    order = sorted(range(N), key=lambda i: file_num(photo_names[i]))
    sorted_names = [photo_names[i] for i in order]
    sorted_dino  = dino_matrix[order]

    print(f'\n  -- DINOv2-only (replicating prior agent best: K=10, sim=0.60) --')
    for K_knn in [5, 10, 20]:
        for sim_t in [0.55, 0.60, 0.65, 0.70]:
            labels = knn_conncomp(sorted_dino, sorted_names, K=K_knn, sim_thresh=sim_t)
            m = evaluate(labels, sorted_names)
            tag = f'SeqKNN-dino K={K_knn:2d} sim={sim_t:.2f}'
            row(tag, m)
        print()

    print(f'\n  -- Fused (best alpha from Section 3) at K=10 --')
    best_alpha_val = best_alpha_agg[3]  # alpha that gave best ARI in Agg sweep
    print(f'     Using best Agg alpha={best_alpha_val:.1f} (+ scanning nearby)')

    best_seq_fused = None
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, best_alpha_val]:
        alpha = round(alpha, 2)
        fused = build_fused(dino_matrix, hist_matrix, alpha=alpha)
        sorted_fused = fused[order]
        for K_knn in [5, 10, 20]:
            for sim_t in [0.55, 0.60, 0.65, 0.70]:
                labels = knn_conncomp(sorted_fused, sorted_names, K=K_knn, sim_thresh=sim_t)
                m = evaluate(labels, sorted_names)
                tag = f'SeqKNN-fused alpha={alpha:.1f} K={K_knn:2d} sim={sim_t:.2f}'
                row(tag, m)
                if best_seq_fused is None or m['purity'] > best_seq_fused[0]:
                    best_seq_fused = (m['purity'], m, tag)
            print()

    # ── Summary ────────────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('SUMMARY — Baselines vs Best Results')
    print('='*80)

    print('\nPrior agent baselines:')
    print(f'  DINOv2-only distance-thresh:     ARI=0.003  clusters=?  purity=?')
    print(f'  Fused(alpha=0.3) dist-thresh t=0.50:  ARI=0.428  clusters=40  pure=18/40')
    print(f'  DINOv2-only SeqKNN K=10 sim=0.60:    purity=0.889')

    print('\nThis run — best by ARI:')
    for label, best in [
        ('Agg forced-K', best_agg),
        ('KMeans forced-K', best_km),
        ('Agg alpha sweep K=20', best_alpha_agg),
        ('KMeans alpha sweep K=20', best_alpha_km),
    ]:
        if best is not None:
            print(f'  {label:<32}  ARI={best[1]["ari"]:.3f}  '
                  f'purity={best[1]["purity"]:.3f}  '
                  f'K={best[1]["n_clusters"]}  '
                  f'cov={best[1]["covered"]}/{N_TRUE_PLAYERS}  '
                  f'cfg={best[2]}')

    if best_seq_fused is not None:
        print(f'  {"SeqKNN fused best":<32}  purity={best_seq_fused[0]:.3f}  '
              f'ARI={best_seq_fused[1]["ari"]:.3f}  '
              f'K={best_seq_fused[1]["n_clusters"]}  '
              f'cfg={best_seq_fused[2]}')

    # Detailed cluster breakdown for best forced-K result
    best_forced_k = max(
        [b for b in [best_agg, best_km] if b is not None],
        key=lambda x: x[0],
    )
    if best_forced_k is not None:
        labels_bfk = best_forced_k[3]
        print(f'\n--- Cluster breakdown for best forced-K result: {best_forced_k[2]} ---')
        cluster_to_players: dict[int, dict[int, int]] = {}
        for i, fname in enumerate(photo_names):
            pid = true_player(fname)
            if pid == 0:
                continue
            cid = int(labels_bfk[i])
            cluster_to_players.setdefault(cid, {}).setdefault(pid, 0)
            cluster_to_players[cid][pid] += 1

        sorted_clusters = sorted(cluster_to_players.keys(),
                                 key=lambda c: sum(cluster_to_players[c].values()), reverse=True)
        print(f'  {"Cluster":>8}  {"Size":>5}  Players')
        n_pure_bfk, n_mixed_bfk = 0, 0
        for cid in sorted_clusters:
            players = cluster_to_players[cid]
            size    = sum(players.values())
            n_pids  = len(players)
            marker  = '[PURE]' if n_pids == 1 else f'[MIXED x{n_pids}]'
            pstr    = '  '.join(f'P{pid}:{cnt}' for pid, cnt in
                                sorted(players.items(), key=lambda x: -x[1]))
            print(f'  C{cid:04d}:  {size:5d}  {pstr}  {marker}')
            if n_pids == 1:
                n_pure_bfk += 1
            else:
                n_mixed_bfk += 1
        print(f'\n  Pure={n_pure_bfk}  Mixed={n_mixed_bfk}  '
              f'Total clusters={len(sorted_clusters)}')

        # Over-segmentation: players split across clusters
        player_to_clusters_bfk: dict[int, set] = {}
        for i, fname in enumerate(photo_names):
            pid = true_player(fname)
            if pid > 0:
                player_to_clusters_bfk.setdefault(pid, set()).add(int(labels_bfk[i]))
        print('\n  Players split across multiple clusters:')
        any_split = False
        for pid in sorted(player_to_clusters_bfk):
            if len(player_to_clusters_bfk[pid]) > 1:
                print(f'    P{pid}: {sorted(player_to_clusters_bfk[pid])}')
                any_split = True
        if not any_split:
            print('    None!')

    print('\n[Done]')


if __name__ == '__main__':
    main()

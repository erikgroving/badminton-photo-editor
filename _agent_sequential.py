"""
Sequential K-nearest-neighbor graph with community detection for player identity clustering.

Tests three variants:
  Variant A: Sequential K-NN graph + connected-components / Louvain community detection
  Variant B: Temporally-weighted cosine similarity + AgglomerativeClustering
  Variant C: Spectral clustering on temporally-weighted similarity matrix

Dataset: 202605 WA Open MS Raws (0001-0696, 20 true players)
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, r'C:\Users\erikg\OneDrive\Desktop\CodePlayground\badminton-photo-editor')
from inference.player_coverage import build_detector, build_embedder, embed_crops, cluster_embeddings
from data.raw_reader import extract_thumbnail

# ── Ground truth ───────────────────────────────────────────────────────────────

RANGES = [
    (1,   78,  1), (79,  107,  2), (108, 126,  3), (127, 150,  2),
    (151, 177,  4), (178, 186,  5), (187, 194,  6), (195, 227,  5),
    (228, 252,  6), (253, 290,  7), (291, 296,  8), (297, 327,  9),
    (328, 347, 10), (348, 401, 11), (402, 442, 12), (443, 473, 13),
    (474, 485, 14), (486, 509, 15), (510, 527, 14), (528, 575, 16),
    (576, 595, 17), (596, 634, 18), (635, 653, 19), (654, 696, 20),
]
N_TRUE_PLAYERS = 20

# Players that appear in two separate, non-consecutive runs (the "gap" case):
#   Player 2:  frames 79-107  AND  127-150
#   Player 5:  frames 178-186 AND  195-227
#   Player 6:  frames 187-194 AND  228-252
#   Player 14: frames 474-485 AND  510-527
GAP_PLAYERS = {2, 5, 6, 14}


def file_num(fname: str) -> int:
    m = re.search(r'(\d{4,})', Path(fname).stem)
    return int(m.group(1)) if m else -1


def true_player(fname: str) -> int:
    n = file_num(fname)
    for s, e, pid in RANGES:
        if s <= n <= e:
            return pid
    return 0  # unlabeled


# ── Dataset loading ────────────────────────────────────────────────────────────

RAW_DIR = Path(r'C:\Users\erikg\Downloads\Jay Ma Photography Training\202605 WA Open MS Raws')
EMBED_CACHE = Path(r'C:\Users\erikg\OneDrive\Desktop\CodePlayground\badminton-photo-editor\cache\_agent_seq_embs.npz')


def load_or_embed() -> tuple[np.ndarray, list[str]]:
    """Return (embeddings [N, 768], filenames [N]).  Uses disk cache if available."""
    if EMBED_CACHE.exists():
        print(f'Loading cached embeddings from {EMBED_CACHE}')
        data = np.load(EMBED_CACHE, allow_pickle=True)
        return data['embs'], list(data['names'])

    _seen: set[str] = set()
    labeled_raws: list[Path] = []
    for p in sorted(RAW_DIR.glob('*'), key=lambda x: file_num(x.name)):
        if p.suffix.lower() != '.cr3' or p.name.lower() in _seen:
            continue
        _seen.add(p.name.lower())
        if 1 <= file_num(p.name) <= 696:
            labeled_raws.append(p)

    print(f'Photos: {len(labeled_raws)}  True players: {N_TRUE_PLAYERS}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    detector = build_detector()
    embedder = build_embedder(device)

    t0 = time.time()
    photo_embs: list[np.ndarray] = []
    photo_names: list[str] = []

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
        largest = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        emb = embed_crops([img.crop(largest)], embedder, device)[0]
        photo_embs.append(emb)
        photo_names.append(p.name)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f'  {i+1}/{len(labeled_raws)} ({len(photo_embs)} detected)  {elapsed:.0f}s')

    embs = np.stack(photo_embs)
    names_arr = np.array(photo_names)
    np.savez(EMBED_CACHE, embs=embs, names=names_arr)
    print(f'Saved embeddings to {EMBED_CACHE}')
    return embs, photo_names


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, names: list[str]) -> dict:
    """
    Returns dict with:
      n_clusters  — number of distinct cluster labels
      coverage    — number of distinct true players with >=1 sample in any cluster
      purity      — fraction of clusters that contain only one true player
      composite   — coverage_frac * purity
      gap_merged  — number of gap players whose two runs ended up in the same cluster
    """
    n_clusters = int(labels.max()) + 1

    cluster_to_players: dict[int, set[int]] = {}
    player_to_clusters: dict[int, set[int]] = {}
    for i, fname in enumerate(names):
        pid = true_player(fname)
        cid = int(labels[i])
        cluster_to_players.setdefault(cid, set()).add(pid)
        player_to_clusters.setdefault(pid, set()).add(cid)

    covered = set(player_to_clusters.keys()) - {0}
    pure = sum(1 for ps in cluster_to_players.values() if len(ps) == 1)
    coverage_frac = len(covered) / N_TRUE_PLAYERS
    purity_frac = pure / n_clusters if n_clusters > 0 else 0.0

    # Gap players: count how many have exactly 1 cluster (their two runs merged)
    gap_merged = sum(
        1 for pid in GAP_PLAYERS
        if pid in player_to_clusters and len(player_to_clusters[pid]) == 1
    )

    return {
        'n_clusters': n_clusters,
        'coverage': len(covered),
        'purity': round(purity_frac, 4),
        'composite': round(coverage_frac * purity_frac, 4),
        'gap_merged': gap_merged,
    }


def print_metrics(label: str, m: dict) -> None:
    print(
        f'  {label:<45}  '
        f'clusters={m["n_clusters"]:4d}  '
        f'coverage={m["coverage"]:2d}/{N_TRUE_PLAYERS}  '
        f'purity={m["purity"]:.3f}  '
        f'composite={m["composite"]:.3f}  '
        f'gap_merged={m["gap_merged"]}/{len(GAP_PLAYERS)}'
    )


# ── Variant A: Sequential K-NN graph + community detection ────────────────────

def variant_a(embs: np.ndarray, names: list[str], K: int, sim_thresh: float, use_louvain: bool = True) -> dict:
    """
    Build a K-NN graph where each node i is connected to its K nearest
    *sequential* neighbours (i-K..i-1, i+1..i+K) with edge weight = cosine sim.
    Only add edge if sim >= sim_thresh.
    Then detect communities via Louvain (or connected-components fallback).
    """
    import networkx as nx

    N = len(embs)
    # Normalise embeddings
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    e = embs / norms

    G = nx.Graph()
    G.add_nodes_from(range(N))

    for i in range(N):
        lo = max(0, i - K)
        hi = min(N, i + K + 1)
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

    labels = np.full(N, -1, dtype=int)

    if use_louvain:
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G, weight='weight', random_state=42)
            for node, cid in partition.items():
                labels[node] = cid
        except ImportError:
            use_louvain = False

    if not use_louvain or (labels == -1).any():
        # Fallback: connected components
        cid = 0
        comp_labels = np.full(N, -1, dtype=int)
        for component in nx.connected_components(G):
            for node in component:
                comp_labels[node] = cid
            cid += 1
        # Isolated nodes (no edges) get unique clusters
        for i in range(N):
            if comp_labels[i] == -1:
                comp_labels[i] = cid
                cid += 1
        labels = comp_labels

    return compute_metrics(labels, names)


# ── Variant B: Temporally-weighted cosine similarity + Agglomerative ──────────

def build_temporal_sim_matrix(embs: np.ndarray, names: list[str], lam: float) -> np.ndarray:
    """
    S[i,j] = cosine_sim(i,j) * exp(-lam * |file_num_i - file_num_j|)
    """
    N = len(embs)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    e = embs / norms
    cos_sim = e @ e.T  # [N, N]
    cos_sim = np.clip(cos_sim, 0.0, 1.0)

    nums = np.array([file_num(n) for n in names], dtype=np.float32)
    gap = np.abs(nums[:, None] - nums[None, :])  # [N, N]
    temporal_w = np.exp(-lam * gap)

    return cos_sim * temporal_w


def variant_b(embs: np.ndarray, names: list[str], lam: float, dist_thresh: float) -> dict:
    """
    Build temporally-weighted similarity matrix, convert to distance, cluster.
    Uses AgglomerativeClustering with average linkage on cosine distance.
    dist_thresh is the AgglomerativeClustering distance_threshold parameter
    (1 - similarity, so higher = fewer clusters).
    """
    from sklearn.cluster import AgglomerativeClustering

    S = build_temporal_sim_matrix(embs, names, lam)
    # Convert similarity to distance (bounded [0,1])
    D = 1.0 - S
    D = np.clip(D, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)

    if len(embs) == 1:
        return compute_metrics(np.array([0]), names)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=dist_thresh,
        metric='precomputed',
        linkage='average',
    )
    labels = clustering.fit_predict(D)
    return compute_metrics(labels, names)


# ── Variant C: Spectral clustering on temporally-weighted similarity ───────────

def variant_c(embs: np.ndarray, names: list[str], lam: float, n_clusters: int) -> dict:
    """
    Build temporally-weighted affinity matrix and apply SpectralClustering.
    """
    from sklearn.cluster import SpectralClustering

    S = build_temporal_sim_matrix(embs, names, lam)
    np.fill_diagonal(S, 1.0)  # self-similarity = 1

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42,
        n_init=10,
    )
    labels = clustering.fit_predict(S)
    return compute_metrics(labels, names)


def variant_c_ari(embs: np.ndarray, names: list[str], lam: float, n_clusters: int) -> float:
    """Return Adjusted Rand Index for spectral clustering variant."""
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import adjusted_rand_score

    S = build_temporal_sim_matrix(embs, names, lam)
    np.fill_diagonal(S, 1.0)

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42,
        n_init=10,
    )
    labels = clustering.fit_predict(S)
    true_labels = np.array([true_player(n) for n in names])
    return adjusted_rand_score(true_labels, labels)


# ── Main grid search ───────────────────────────────────────────────────────────

def main() -> None:
    embs, names = load_or_embed()
    print(f'\nEmbeddings: {embs.shape}  Photos with detections: {len(names)}')

    # ── Variant A ──────────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('VARIANT A: Sequential K-NN graph + Louvain community detection')
    print('='*80)
    print(f'  {"Config":<45}  {"clusters":>8}  {"cov":>6}  {"purity":>7}  {"composite":>10}  {"gap":>5}')

    best_a: Optional[tuple[float, dict, str]] = None
    K_values = [3, 5, 10, 20]
    thresh_values = [0.30, 0.40, 0.50, 0.60]

    for K in K_values:
        for thresh in thresh_values:
            tag = f'K={K:2d} sim_thresh={thresh:.2f} (Louvain)'
            m = variant_a(embs, names, K=K, sim_thresh=thresh, use_louvain=True)
            print_metrics(tag, m)
            if best_a is None or m['composite'] > best_a[0]:
                best_a = (m['composite'], m, tag)
        print()

    # Also try connected-components (no Louvain)
    print('  -- Connected-components fallback --')
    for K in K_values:
        for thresh in thresh_values:
            tag = f'K={K:2d} sim_thresh={thresh:.2f} (ConnComp)'
            m = variant_a(embs, names, K=K, sim_thresh=thresh, use_louvain=False)
            print_metrics(tag, m)
            if best_a is None or m['composite'] > best_a[0]:
                best_a = (m['composite'], m, tag)
        print()

    print(f'\nBest Variant A: {best_a[2]}')
    print(f'  {best_a[1]}')

    # ── Variant B ──────────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('VARIANT B: Temporally-weighted cosine similarity + AgglomerativeClustering')
    print('='*80)
    print(f'  {"Config":<45}  {"clusters":>8}  {"cov":>6}  {"purity":>7}  {"composite":>10}  {"gap":>5}')

    best_b: Optional[tuple[float, dict, str]] = None
    lambda_values = [0.01, 0.05, 0.10, 0.20]
    dist_thresh_values = [0.30, 0.40, 0.50, 0.60]

    for lam in lambda_values:
        for dt in dist_thresh_values:
            tag = f'lam={lam:.2f} dist_thresh={dt:.2f}'
            m = variant_b(embs, names, lam=lam, dist_thresh=dt)
            print_metrics(tag, m)
            if best_b is None or m['composite'] > best_b[0]:
                best_b = (m['composite'], m, tag)
        print()

    print(f'\nBest Variant B: {best_b[2]}')
    print(f'  {best_b[1]}')

    # ── Variant C ──────────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('VARIANT C: Spectral clustering on temporally-weighted similarity matrix')
    print('='*80)
    print(f'  {"Config":<45}  {"clusters":>8}  {"cov":>6}  {"purity":>7}  {"composite":>10}  {"gap":>5}')

    best_c: Optional[tuple[float, dict, str]] = None
    lambda_values_c = [0.01, 0.05, 0.10, 0.20]
    n_clusters_values = [N_TRUE_PLAYERS]  # 20 known; also try auto via different k

    for lam in lambda_values_c:
        for nc in n_clusters_values:
            tag = f'lam={lam:.2f} n_clusters={nc}'
            m = variant_c(embs, names, lam=lam, n_clusters=nc)
            print_metrics(tag, m)
            if best_c is None or m['composite'] > best_c[0]:
                best_c = (m['composite'], m, tag)
        print()

    # Also try 'auto' via nearby cluster counts
    print('  -- Auto n_clusters sweep --')
    best_lam_c = float(best_c[2].split('lam=')[1].split(' ')[0]) if best_c else 0.05
    for nc in [15, 18, 20, 22, 25, 30]:
        tag = f'lam={best_lam_c:.2f} n_clusters={nc} (sweep)'
        m = variant_c(embs, names, lam=best_lam_c, n_clusters=nc)
        print_metrics(tag, m)
        if best_c is None or m['composite'] > best_c[0]:
            best_c = (m['composite'], m, tag)
    print()

    print(f'\nBest Variant C: {best_c[2]}')
    print(f'  {best_c[1]}')

    # ARI for best Variant C config
    try:
        best_lam_c_final = float(best_c[2].split('lam=')[1].split(' ')[0])
        best_nc_final = int(best_c[2].split('n_clusters=')[1].split(' ')[0].rstrip(')'))
        ari = variant_c_ari(embs, names, lam=best_lam_c_final, n_clusters=best_nc_final)
        print(f'  Adjusted Rand Index (best C config, n_clusters={best_nc_final}): {ari:.4f}')
    except Exception as ex:
        print(f'  ARI computation failed: {ex}')

    # ── Summary ────────────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('SUMMARY')
    print('='*80)

    variants = [
        ('A (Sequential K-NN + Louvain)', best_a),
        ('B (Temporal-weighted + Agglomerative)', best_b),
        ('C (Spectral on temporal affinity)', best_c),
    ]
    ranked = sorted(variants, key=lambda x: x[1][0], reverse=True)

    print('\nRanked by composite score (coverage_frac * purity):')
    for rank, (name, (score, m, cfg)) in enumerate(ranked, 1):
        print(f'  #{rank} Variant {name}')
        print(f'       Config: {cfg}')
        print(f'       clusters={m["n_clusters"]}  coverage={m["coverage"]}/{N_TRUE_PLAYERS}  '
              f'purity={m["purity"]:.3f}  composite={score:.4f}  '
              f'gap_merged={m["gap_merged"]}/{len(GAP_PLAYERS)}')
        print()

    print('Gap-player analysis (players 2, 5, 6, 14 appear in two non-consecutive runs):')
    print('  gap_merged=4 means all 4 "gap" players were correctly identified as the')
    print('  same person across their two separate appearance windows.')
    print()

    # Which variant best handles gap players?
    gap_scores = [
        ('A', best_a[1]['gap_merged'], best_a[0]),
        ('B', best_b[1]['gap_merged'], best_b[0]),
        ('C', best_c[1]['gap_merged'], best_c[0]),
    ]
    gap_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    print(f'Best for gap-player merging: Variant {gap_scores[0][0]} '
          f'(gap_merged={gap_scores[0][1]}/{len(GAP_PLAYERS)})')

    print('\nDone.')


if __name__ == '__main__':
    main()

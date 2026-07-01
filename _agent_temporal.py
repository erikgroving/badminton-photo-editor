"""
Temporal changepoint segmentation parameter sweep.
Tests both running-average and hard-reset (compare to segment start) variants.
Evaluates on WA Open (696 photos, 20 players) and Demo (171 photos, 5 players).
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r'C:\Users\erikg\OneDrive\Desktop\CodePlayground\badminton-photo-editor')
from inference.player_coverage import build_detector, build_embedder, embed_crops, cluster_embeddings
from data.raw_reader import extract_thumbnail

# ── Ground truth: WA Open ─────────────────────────────────────────────────────
WA_RANGES = [
    (1,   78,  1), (79,  107,  2), (108, 126,  3), (127, 150,  2),
    (151, 177,  4), (178, 186,  5), (187, 194,  6), (195, 227,  5),
    (228, 252,  6), (253, 290,  7), (291, 296,  8), (297, 327,  9),
    (328, 347, 10), (348, 401, 11), (402, 442, 12), (443, 473, 13),
    (474, 485, 14), (486, 509, 15), (510, 527, 14), (528, 575, 16),
    (576, 595, 17), (596, 634, 18), (635, 653, 19), (654, 696, 20),
]
WA_N_PLAYERS = 20

# ── Ground truth: Demo ────────────────────────────────────────────────────────
DEMO_GROUPS = {
    1: set(range(1316, 1344)),
    2: set(range(1344, 1390)),
    3: set(range(1390, 1447)),
    4: set(range(1447, 1480)),
    5: set(range(1480, 1487)),
}
DEMO_N_PLAYERS = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def file_num(fname: str) -> int:
    m = re.search(r'(\d{4,})', Path(fname).stem)
    return int(m.group(1)) if m else -1


def wa_true_player(fname: str) -> int:
    n = file_num(fname)
    for s, e, pid in WA_RANGES:
        if s <= n <= e:
            return pid
    return 0


def demo_true_player(fname: str) -> int:
    n = file_num(fname)
    for g, rng in DEMO_GROUPS.items():
        if n in rng:
            return g
    return 0


# ── Embedding extraction ──────────────────────────────────────────────────────

def embed_dataset(raw_paths: list[Path], detector, embedder, device, label: str):
    """Return (names, embs) lists — one entry per photo with a detected person."""
    names, embs = [], []
    n = len(raw_paths)
    t0 = time.time()
    for i, p in enumerate(raw_paths):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f'  [{label}] {i+1}/{n}  detected={len(names)}  '
                  f'elapsed={elapsed:.0f}s  eta={eta:.0f}s', flush=True)
        try:
            img = extract_thumbnail(p, size=(448, 448))
        except Exception as ex:
            print(f'  WARNING: extract_thumbnail failed for {p.name}: {ex}')
            continue
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
        if best_box is None:
            continue
        crop = img.crop(best_box)
        try:
            emb = embed_crops([crop], embedder, device)[0]
        except Exception as ex:
            print(f'  WARNING: embed failed for {p.name}: {ex}')
            continue
        names.append(p.name)
        embs.append(emb)
    print(f'  [{label}] Done: {len(names)}/{n} detected in {time.time()-t0:.0f}s')
    return names, embs


# ── Segmentation algorithms ───────────────────────────────────────────────────

def segment_running_avg(names, embs, change_thresh: float):
    """Standard: compare each photo to running average of current segment."""
    order = sorted(range(len(names)), key=lambda i: file_num(names[i]))
    names = [names[i] for i in order]
    embs  = [embs[i]  for i in order]

    segments = []  # list of (list_of_names, avg_embedding)
    seg_names = [names[0]]
    seg_embs  = [embs[0]]
    seg_avg   = embs[0] / (np.linalg.norm(embs[0]) + 1e-8)

    for i in range(1, len(names)):
        e = embs[i] / (np.linalg.norm(embs[i]) + 1e-8)
        dist = 1.0 - float(np.dot(seg_avg, e))
        if dist > change_thresh:
            avg = np.mean(seg_embs, axis=0)
            avg /= np.linalg.norm(avg) + 1e-8
            segments.append((seg_names, avg))
            seg_names = [names[i]]
            seg_embs  = [embs[i]]
            seg_avg   = e.copy()
        else:
            seg_names.append(names[i])
            seg_embs.append(embs[i])
            seg_avg = np.mean(seg_embs, axis=0)
            seg_avg /= np.linalg.norm(seg_avg) + 1e-8

    avg = np.mean(seg_embs, axis=0)
    avg /= np.linalg.norm(avg) + 1e-8
    segments.append((seg_names, avg))
    return segments


def segment_hard_reset(names, embs, change_thresh: float):
    """Hard-reset: compare each photo to the FIRST photo of current segment."""
    order = sorted(range(len(names)), key=lambda i: file_num(names[i]))
    names = [names[i] for i in order]
    embs  = [embs[i]  for i in order]

    segments = []
    seg_names = [names[0]]
    seg_embs  = [embs[0]]
    seg_start = embs[0] / (np.linalg.norm(embs[0]) + 1e-8)  # anchor = first photo

    for i in range(1, len(names)):
        e = embs[i] / (np.linalg.norm(embs[i]) + 1e-8)
        dist = 1.0 - float(np.dot(seg_start, e))
        if dist > change_thresh:
            avg = np.mean(seg_embs, axis=0)
            avg /= np.linalg.norm(avg) + 1e-8
            segments.append((seg_names, avg))
            seg_names = [names[i]]
            seg_embs  = [embs[i]]
            seg_start = e.copy()
        else:
            seg_names.append(names[i])
            seg_embs.append(embs[i])

    avg = np.mean(seg_embs, axis=0)
    avg /= np.linalg.norm(avg) + 1e-8
    segments.append((seg_names, avg))
    return segments


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(segments, cluster_thresh: float, true_player_fn, n_true_players: int):
    """Cluster segment embeddings and compute coverage + purity metrics."""
    n_segs = len(segments)
    seg_matrix = np.stack([s[1] for s in segments])

    if n_segs == 1:
        labels = np.array([0])
    else:
        labels = cluster_embeddings(seg_matrix, cluster_thresh)

    n_clusters = int(labels.max()) + 1

    # Build cluster -> set of true players
    cluster_to_players: dict[int, set] = {}
    for seg_idx, (seg_names, _) in enumerate(segments):
        cid = int(labels[seg_idx])
        for fname in seg_names:
            pid = true_player_fn(fname)
            if pid != 0:
                cluster_to_players.setdefault(cid, set()).add(pid)

    # Coverage: fraction of true players with >=1 cluster
    covered_players = set()
    for pids in cluster_to_players.values():
        covered_players |= pids
    covered_players.discard(0)
    coverage = len(covered_players) / n_true_players

    # Purity: fraction of clusters containing only ONE true player
    pure = sum(1 for pids in cluster_to_players.values() if len(pids) == 1)
    purity = pure / n_clusters if n_clusters > 0 else 0.0

    return {
        'n_segs': n_segs,
        'n_clusters': n_clusters,
        'coverage': coverage,
        'coverage_n': len(covered_players),
        'purity': purity,
        'score': coverage * purity,
        'covered_players': covered_players,
    }


# ── Grid sweep ────────────────────────────────────────────────────────────────

CHANGE_THRESHOLDS  = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
CLUSTER_THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def run_sweep(names, embs, true_player_fn, n_true_players, label):
    print(f'\n{"="*70}')
    print(f'SWEEP: {label}  (n_photos={len(names)}, n_true_players={n_true_players})')
    print(f'{"="*70}')

    results_avg  = []
    results_hard = []

    for ct in CHANGE_THRESHOLDS:
        segs_avg  = segment_running_avg(names, embs, ct)
        segs_hard = segment_hard_reset(names, embs, ct)
        for ht in CLUSTER_THRESHOLDS:
            m_avg  = evaluate(segs_avg,  ht, true_player_fn, n_true_players)
            m_hard = evaluate(segs_hard, ht, true_player_fn, n_true_players)
            results_avg.append( {'change_thresh': ct, 'cluster_thresh': ht, 'variant': 'avg',  **m_avg})
            results_hard.append({'change_thresh': ct, 'cluster_thresh': ht, 'variant': 'hard', **m_hard})

    all_results = results_avg + results_hard
    all_results.sort(key=lambda r: (-r['score'], -r['coverage'], -r['purity']))

    print(f'\n--- Top 15 configurations by composite score (coverage_frac * purity) ---')
    print(f'{"variant":>6} {"change_t":>9} {"cluster_t":>10} {"n_segs":>7} {"n_clust":>8} '
          f'{"coverage":>9} {"purity":>8} {"score":>7}')
    print('-' * 70)
    for r in all_results[:15]:
        cov_str = f'{r["coverage_n"]}/{n_true_players}'
        print(f'{r["variant"]:>6} {r["change_thresh"]:>9.2f} {r["cluster_thresh"]:>10.2f} '
              f'{r["n_segs"]:>7d} {r["n_clusters"]:>8d} '
              f'{cov_str:>9} {r["purity"]:>8.3f} {r["score"]:>7.3f}')

    print(f'\n--- Top 10: Running-average variant ---')
    top_avg = sorted(results_avg, key=lambda r: (-r['score'], -r['coverage'], -r['purity']))[:10]
    print(f'{"change_t":>9} {"cluster_t":>10} {"n_segs":>7} {"n_clust":>8} '
          f'{"coverage":>9} {"purity":>8} {"score":>7}')
    print('-' * 60)
    for r in top_avg:
        cov_str = f'{r["coverage_n"]}/{n_true_players}'
        print(f'{r["change_thresh"]:>9.2f} {r["cluster_thresh"]:>10.2f} '
              f'{r["n_segs"]:>7d} {r["n_clusters"]:>8d} '
              f'{cov_str:>9} {r["purity"]:>8.3f} {r["score"]:>7.3f}')

    print(f'\n--- Top 10: Hard-reset variant ---')
    top_hard = sorted(results_hard, key=lambda r: (-r['score'], -r['coverage'], -r['purity']))[:10]
    print(f'{"change_t":>9} {"cluster_t":>10} {"n_segs":>7} {"n_clust":>8} '
          f'{"coverage":>9} {"purity":>8} {"score":>7}')
    print('-' * 60)
    for r in top_hard:
        cov_str = f'{r["coverage_n"]}/{n_true_players}'
        print(f'{r["change_thresh"]:>9.2f} {r["cluster_thresh"]:>10.2f} '
              f'{r["n_segs"]:>7d} {r["n_clusters"]:>8d} '
              f'{cov_str:>9} {r["purity"]:>8.3f} {r["score"]:>7.3f}')

    best = all_results[0]
    print(f'\n*** BEST for {label}: variant={best["variant"]} '
          f'change_thresh={best["change_thresh"]:.2f} '
          f'cluster_thresh={best["cluster_thresh"]:.2f}  '
          f'coverage={best["coverage_n"]}/{n_true_players} ({best["coverage"]:.1%})  '
          f'purity={best["purity"]:.3f}  score={best["score"]:.3f}  '
          f'n_segs={best["n_segs"]}  n_clusters={best["n_clusters"]} ***')

    # Find best with FULL coverage
    full_cov = [r for r in all_results if r['coverage_n'] == n_true_players]
    if full_cov:
        best_full = full_cov[0]
        print(f'*** BEST with full coverage: variant={best_full["variant"]} '
              f'change_thresh={best_full["change_thresh"]:.2f} '
              f'cluster_thresh={best_full["cluster_thresh"]:.2f}  '
              f'purity={best_full["purity"]:.3f}  score={best_full["score"]:.3f}  '
              f'n_segs={best_full["n_segs"]}  n_clusters={best_full["n_clusters"]} ***')
    else:
        print(f'  No configuration achieves full coverage on {label}.')
        best_cov = max(all_results, key=lambda r: (r['coverage_n'], r['purity']))
        print(f'  Best partial: variant={best_cov["variant"]} '
              f'change_thresh={best_cov["change_thresh"]:.2f} '
              f'cluster_thresh={best_cov["cluster_thresh"]:.2f}  '
              f'coverage={best_cov["coverage_n"]}/{n_true_players}  '
              f'purity={best_cov["purity"]:.3f}')

    return all_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\nBuilding detector + embedder...')
    detector = build_detector()
    embedder = build_embedder(device)
    print('Models ready.')

    # ── WA Open dataset ───────────────────────────────────────────────────────
    wa_dir = Path(r'C:\Users\erikg\Downloads\Jay Ma Photography Training\202605 WA Open MS Raws')
    _seen = set()
    wa_raws = []
    for p in sorted(wa_dir.glob('*.CR3'), key=lambda x: file_num(x.name)):
        if p.name.lower() in _seen:
            continue
        _seen.add(p.name.lower())
        n = file_num(p.name)
        if 1 <= n <= 696:
            wa_raws.append(p)
    # Also check lowercase extension
    for p in sorted(wa_dir.glob('*.cr3'), key=lambda x: file_num(x.name)):
        if p.name.lower() in _seen:
            continue
        _seen.add(p.name.lower())
        n = file_num(p.name)
        if 1 <= n <= 696:
            wa_raws.append(p)
    wa_raws.sort(key=lambda x: file_num(x.name))
    print(f'\nWA Open: {len(wa_raws)} labeled photos (0001-0696, 20 true players)')

    print('\nEmbedding WA Open...')
    wa_names, wa_embs = embed_dataset(wa_raws, detector, embedder, device, 'WA Open')
    print(f'WA Open detected: {len(wa_names)}/{len(wa_raws)}')

    # ── Demo dataset ──────────────────────────────────────────────────────────
    demo_dir = Path(r'C:\Users\erikg\Downloads\Demo')
    demo_raws = []
    _seen2 = set()
    if demo_dir.exists():
        for p in sorted(demo_dir.glob('*.CR3'), key=lambda x: file_num(x.name)):
            if p.name.lower() not in _seen2:
                _seen2.add(p.name.lower())
                demo_raws.append(p)
        for p in sorted(demo_dir.glob('*.cr3'), key=lambda x: file_num(x.name)):
            if p.name.lower() not in _seen2:
                _seen2.add(p.name.lower())
                demo_raws.append(p)
        demo_raws.sort(key=lambda x: file_num(x.name))
        print(f'\nDemo: {len(demo_raws)} photos (5 true players)')
        print('\nEmbedding Demo...')
        demo_names, demo_embs = embed_dataset(demo_raws, detector, embedder, device, 'Demo')
        print(f'Demo detected: {len(demo_names)}/{len(demo_raws)}')
    else:
        print(f'\nDemo directory not found at {demo_dir} — skipping.')
        demo_names, demo_embs = [], []

    # ── Run sweeps ────────────────────────────────────────────────────────────
    print('\n\n' + '#'*70)
    print('# PARAMETER SWEEP RESULTS')
    print('#'*70)

    wa_results = run_sweep(wa_names, wa_embs, wa_true_player, WA_N_PLAYERS, 'WA Open (20 players)')

    if demo_names:
        demo_results = run_sweep(demo_names, demo_embs, demo_true_player, DEMO_N_PLAYERS, 'Demo (5 players)')

    # ── Cross-dataset summary ─────────────────────────────────────────────────
    if demo_names:
        print('\n\n' + '#'*70)
        print('# CROSS-DATASET SUMMARY')
        print('#'*70)

        # Build dicts keyed by (variant, change_thresh, cluster_thresh)
        wa_dict  = {(r['variant'], r['change_thresh'], r['cluster_thresh']): r for r in wa_results}
        demo_dict = {(r['variant'], r['change_thresh'], r['cluster_thresh']): r for r in demo_results}

        combined = []
        for key, wr in wa_dict.items():
            dr = demo_dict.get(key)
            if dr:
                combo_score = (wr['score'] + dr['score']) / 2
                combined.append({
                    'key': key,
                    'wa_coverage':  wr['coverage_n'],
                    'wa_purity':    wr['purity'],
                    'wa_score':     wr['score'],
                    'demo_coverage': dr['coverage_n'],
                    'demo_purity':  dr['purity'],
                    'demo_score':   dr['score'],
                    'combo_score':  combo_score,
                })
        combined.sort(key=lambda x: -x['combo_score'])

        print(f'\n--- Top 10 by average score across both datasets ---')
        print(f'{"variant":>6} {"ct":>5} {"ht":>5}  '
              f'{"WA cov":>8} {"WA pur":>7} {"WA sc":>6}  '
              f'{"D cov":>7} {"D pur":>7} {"D sc":>6}  {"avg":>6}')
        print('-' * 80)
        for c in combined[:10]:
            v, ct, ht = c['key']
            print(f'{v:>6} {ct:>5.2f} {ht:>5.2f}  '
                  f'{c["wa_coverage"]:>3}/{WA_N_PLAYERS}     {c["wa_purity"]:>5.3f}  {c["wa_score"]:>5.3f}  '
                  f'{c["demo_coverage"]:>2}/{DEMO_N_PLAYERS}    {c["demo_purity"]:>5.3f}  {c["demo_score"]:>5.3f}  '
                  f'{c["combo_score"]:>6.3f}')

        best_combo = combined[0]
        v, ct, ht = best_combo['key']
        print(f'\n*** BEST cross-dataset: variant={v} change_thresh={ct:.2f} cluster_thresh={ht:.2f}')
        print(f'    WA Open:  coverage={best_combo["wa_coverage"]}/{WA_N_PLAYERS}  '
              f'purity={best_combo["wa_purity"]:.3f}  score={best_combo["wa_score"]:.3f}')
        print(f'    Demo:     coverage={best_combo["demo_coverage"]}/{DEMO_N_PLAYERS}  '
              f'purity={best_combo["demo_purity"]:.3f}  score={best_combo["demo_score"]:.3f}')
        print(f'    Combined avg score: {best_combo["combo_score"]:.3f} ***')

        # Full coverage on both
        full_both = [c for c in combined
                     if c['wa_coverage'] == WA_N_PLAYERS and c['demo_coverage'] == DEMO_N_PLAYERS]
        if full_both:
            best_full_both = full_both[0]
            v, ct, ht = best_full_both['key']
            print(f'\n*** BEST with FULL COVERAGE on BOTH: variant={v} '
                  f'change_thresh={ct:.2f} cluster_thresh={ht:.2f}')
            print(f'    WA Open:  coverage=20/20  purity={best_full_both["wa_purity"]:.3f}  '
                  f'score={best_full_both["wa_score"]:.3f}')
            print(f'    Demo:     coverage=5/5    purity={best_full_both["demo_purity"]:.3f}  '
                  f'score={best_full_both["demo_score"]:.3f}')
            print(f'    Combined avg score: {best_full_both["combo_score"]:.3f} ***')
        else:
            print('\n  No single config achieves full coverage on BOTH datasets simultaneously.')

    print('\nDone.')


if __name__ == '__main__':
    main()

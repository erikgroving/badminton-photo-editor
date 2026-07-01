"""
EXIF Burst-Level Grouping for Player Identification
====================================================
Groups CR3 photos by EXIF timestamp gaps (bursts), averages embeddings within
each burst, then clusters burst embeddings to identify distinct players.

Falls back to fixed-window grouping if EXIF timestamps are unavailable or uniform.

Primary dataset: 202605 WA Open MS Raws (0V2A0001-0V2A0696, 696 photos, 20 true players)
Secondary: Demo (171 photos, 5 players)
"""
from __future__ import annotations

import io
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, r'C:\Users\erikg\OneDrive\Desktop\CodePlayground\badminton-photo-editor')

import numpy as np
import torch

from data.raw_reader import extract_thumbnail
from inference.player_coverage import (
    build_detector,
    build_embedder,
    cluster_embeddings,
    embed_crops,
)

# ── Ground truth ───────────────────────────────────────────────────────────────

WA_RANGES = [
    (1,    78,   1),  (79,   107,  2),  (108,  126,  3),  (127,  150,  2),
    (151,  177,  4),  (178,  186,  5),  (187,  194,  6),  (195,  227,  5),
    (228,  252,  6),  (253,  290,  7),  (291,  296,  8),  (297,  327,  9),
    (328,  347,  10), (348,  401,  11), (402,  442,  12), (443,  473,  13),
    (474,  485,  14), (486,  509,  15), (510,  527,  14), (528,  575,  16),
    (576,  595,  17), (596,  634,  18), (635,  653,  19), (654,  696,  20),
]
WA_N_PLAYERS = 20

DEMO_GROUPS = {
    1: set(range(1316, 1344)),
    2: set(range(1344, 1390)),
    3: set(range(1390, 1447)),
    4: set(range(1447, 1480)),
    5: set(range(1480, 1487)),
}
DEMO_N_PLAYERS = 5

# ── Helpers ────────────────────────────────────────────────────────────────────

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


# ── EXIF reading ───────────────────────────────────────────────────────────────

def read_exif_timestamp(path: Path) -> float | None:
    """
    Read a precise timestamp from a CR3 file.
    Returns float seconds-since-epoch, or None if unavailable.

    CR3 (Canon Raw v3) uses the CRAW2 container — standard EXIF parsers
    (exifread, Pillow via embedded JPEG) cannot reliably parse it.
    The most reliable method is a raw byte-scan for the ASCII datetime string
    'YYYY:MM:DD HH:MM:SS' embedded in the file header.

    SubSecTimeOriginal is not present in these CR3s; timestamps have 1-second
    resolution. Multiple burst shots therefore share the same whole-second ts.

    Fallback chain:
      1. Raw bytes scan (primary — works on Canon CR3)
      2. exifread (works on standard TIFF-based raws)
      3. rawpy extract_thumb + Pillow EXIF
    """
    import re as _re

    # Method 1: Raw bytes scan — works on Canon CR3
    try:
        with open(path, 'rb') as f:
            # CR3 datetime is within the first ~16 KB of the file header
            data = f.read(65536)
        matches = _re.findall(rb'20\d\d:\d\d:\d\d \d\d:\d\d:\d\d', data)
        if matches:
            dt_str = matches[0].decode()
            dt = datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
            return dt.timestamp()
    except Exception:
        pass

    # Method 2: exifread directly on CR3 (works on standard TIFF-based raws)
    try:
        import exifread
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False, stop_tag='EXIF SubSecTimeOriginal')
        dto = tags.get('EXIF DateTimeOriginal') or tags.get('Image DateTimeOriginal')
        sub = tags.get('EXIF SubSecTimeOriginal') or tags.get('EXIF SubSecTime')
        if dto:
            dt_str = str(dto).strip()
            try:
                dt = datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
                ts = dt.timestamp()
                if sub:
                    sub_str = str(sub).strip()
                    if sub_str.isdigit():
                        ts += float('0.' + sub_str)
                return ts
            except ValueError:
                pass
    except Exception:
        pass

    # Method 3: rawpy + embedded JPEG + Pillow EXIF
    try:
        import rawpy
        from PIL import Image, ExifTags

        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()

        if thumb.format == rawpy.ThumbFormat.JPEG:
            img = Image.open(io.BytesIO(thumb.data))
            exif_data = img._getexif()
            if exif_data:
                tag_map = {v: k for k, v in ExifTags.TAGS.items()}
                dto_tag   = tag_map.get('DateTimeOriginal')
                sub_tag   = tag_map.get('SubSecTimeOriginal') or tag_map.get('SubSecTime')
                dto_val   = exif_data.get(dto_tag) if dto_tag else None
                sub_val   = exif_data.get(sub_tag) if sub_tag else None
                if dto_val:
                    dt = datetime.strptime(str(dto_val).strip(), '%Y:%m:%d %H:%M:%S')
                    ts = dt.timestamp()
                    if sub_val:
                        sv = str(sub_val).strip()
                        if sv.isdigit():
                            ts += float('0.' + sv)
                    return ts
    except Exception:
        pass

    return None


# ── Embedding pass ─────────────────────────────────────────────────────────────

def embed_photos(
    raw_paths: list[Path],
    detector,
    embedder,
    device: torch.device,
    thumb_size: tuple[int, int] = (448, 448),
) -> tuple[dict[str, np.ndarray], dict[str, float | None]]:
    """
    For each path: extract thumbnail, detect largest person, embed.
    Returns:
      emb_by_fname: fname -> 768-d numpy embedding (only for detected photos)
      ts_by_fname:  fname -> float timestamp or None
    """
    emb_by_fname: dict[str, np.ndarray] = {}
    ts_by_fname:  dict[str, float | None] = {}

    n = len(raw_paths)
    for i, p in enumerate(raw_paths):
        fname = p.name
        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{n}] detecting + embedding...')

        # Timestamp
        ts_by_fname[fname] = read_exif_timestamp(p)

        # Detection + embedding
        try:
            img = extract_thumbnail(p, size=thumb_size)
        except Exception:
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
            embs = embed_crops([crop], embedder, device)
            emb_by_fname[fname] = embs[0]
        except Exception:
            pass

    return emb_by_fname, ts_by_fname


# ── Burst grouping strategies ──────────────────────────────────────────────────

def group_by_exif_gap(
    fnames_sorted: list[str],
    ts_by_fname: dict[str, float | None],
    gap_thresh: float,
) -> list[list[str]]:
    """
    Group sorted filenames into bursts: new burst when timestamp gap > gap_thresh seconds.
    Photos with no timestamp are assigned to the current burst.
    """
    bursts: list[list[str]] = []
    current: list[str] = []
    last_ts: float | None = None

    for fname in fnames_sorted:
        ts = ts_by_fname.get(fname)
        if ts is None:
            # No timestamp — attach to current burst
            if not current:
                current = [fname]
            else:
                current.append(fname)
        else:
            if last_ts is None or (ts - last_ts) <= gap_thresh:
                current.append(fname)
            else:
                if current:
                    bursts.append(current)
                current = [fname]
            last_ts = ts

    if current:
        bursts.append(current)
    return bursts


def group_by_fixed_window(
    fnames_sorted: list[str],
    window_size: int,
) -> list[list[str]]:
    """Group sorted filenames into non-overlapping windows of fixed size."""
    bursts = []
    for i in range(0, len(fnames_sorted), window_size):
        bursts.append(fnames_sorted[i:i + window_size])
    return bursts


def group_no_grouping(fnames_sorted: list[str]) -> list[list[str]]:
    """Each photo is its own 'burst'."""
    return [[f] for f in fnames_sorted]


# ── Burst embedding + clustering ───────────────────────────────────────────────

def burst_embeddings(
    bursts: list[list[str]],
    emb_by_fname: dict[str, np.ndarray],
) -> tuple[list[list[str]], np.ndarray]:
    """
    For each burst with at least one detected photo, average the embeddings.
    Returns (valid_bursts, burst_emb_matrix).
    """
    valid_bursts: list[list[str]] = []
    burst_avgs: list[np.ndarray] = []

    for burst in bursts:
        embs = [emb_by_fname[f] for f in burst if f in emb_by_fname]
        if not embs:
            continue
        arr = np.stack(embs)
        avg = arr.mean(axis=0)
        avg /= np.linalg.norm(avg) + 1e-8
        valid_bursts.append(burst)
        burst_avgs.append(avg)

    if not burst_avgs:
        return [], np.zeros((0, 768), dtype=np.float32)

    return valid_bursts, np.stack(burst_avgs)


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_metrics(
    bursts: list[list[str]],
    burst_matrix: np.ndarray,
    cluster_thresh: float,
    true_player_fn,
    n_true_players: int,
) -> dict:
    """
    Cluster burst embeddings and compute:
      n_bursts, n_clusters, coverage (frac of true players covered),
      purity (frac of clusters that are single-player), composite score.
    """
    if len(burst_matrix) == 0:
        return {
            'n_bursts': 0, 'n_clusters': 0,
            'coverage_count': 0, 'coverage_frac': 0.0,
            'purity': 0.0, 'composite': 0.0,
        }

    labels = cluster_embeddings(burst_matrix, cluster_thresh)
    n_clusters = int(labels.max()) + 1

    # Map each cluster to the set of true players in its bursts
    cluster_to_players: dict[int, set[int]] = {}
    for burst_idx, (burst_fnames, cid) in enumerate(zip(bursts, labels)):
        cid_int = int(cid)
        for fname in burst_fnames:
            pid = true_player_fn(fname)
            if pid != 0:
                cluster_to_players.setdefault(cid_int, set()).add(pid)

    # Coverage: which true players appear in at least one cluster
    covered_players: set[int] = set()
    for players in cluster_to_players.values():
        covered_players.update(players)

    # Purity: fraction of non-empty clusters with exactly one player
    non_empty = [ps for ps in cluster_to_players.values() if ps]
    pure_count = sum(1 for ps in non_empty if len(ps) == 1)
    purity = pure_count / len(non_empty) if non_empty else 0.0

    coverage_frac = len(covered_players) / n_true_players

    return {
        'n_bursts':       len(bursts),
        'n_clusters':     n_clusters,
        'coverage_count': len(covered_players),
        'coverage_frac':  coverage_frac,
        'purity':         purity,
        'composite':      coverage_frac * purity,
    }


# ── 20%-kept simulation ────────────────────────────────────────────────────────

def simulate_culled(
    fnames_sorted: list[str],
    true_player_fn,
    n_true_players: int,
    keep_frac: float = 0.20,
    seed: int = 42,
) -> list[str]:
    """
    Simulate a cull: keep ~keep_frac of photos per player, randomly.
    Returns sorted list of surviving filenames.
    """
    rng = random.Random(seed)
    # Group by player
    player_to_fnames: dict[int, list[str]] = {}
    no_player: list[str] = []
    for fname in fnames_sorted:
        pid = true_player_fn(fname)
        if pid == 0:
            no_player.append(fname)
        else:
            player_to_fnames.setdefault(pid, []).append(fname)

    kept: list[str] = list(no_player)  # keep unlabeled (shouldn't affect WA test)
    for pid, pnames in player_to_fnames.items():
        k = max(1, int(len(pnames) * keep_frac))
        kept.extend(rng.sample(pnames, k))

    kept_set = set(kept)
    return [f for f in fnames_sorted if f in kept_set]


# ── Main ───────────────────────────────────────────────────────────────────────

def run_dataset(
    name: str,
    raw_dir: Path,
    raw_paths: list[Path],
    true_player_fn,
    n_true_players: int,
    detector,
    embedder,
    device: torch.device,
) -> None:
    sep = '=' * 70
    print(f'\n{sep}')
    print(f'DATASET: {name}')
    print(f'Photos: {len(raw_paths)}  |  True players: {n_true_players}')
    print(sep)

    # ── Step 1: Embed all photos ──────────────────────────────────────────────
    print('\nStep 1: Extracting thumbnails, detecting persons, embedding...')
    t0 = time.time()
    emb_by_fname, ts_by_fname = embed_photos(raw_paths, detector, embedder, device)
    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.1f}s  |  Detected: {len(emb_by_fname)}/{len(raw_paths)}')

    # ── Step 2: Analyse timestamps ────────────────────────────────────────────
    print('\nStep 2: EXIF timestamp analysis...')
    all_ts = [v for v in ts_by_fname.values() if v is not None]
    ts_available = len(all_ts)
    ts_unique = len(set(all_ts))
    print(f'  Timestamps available:  {ts_available}/{len(raw_paths)}')
    print(f'  Unique timestamps:     {ts_unique}')

    if ts_available > 1:
        sorted_ts = sorted(all_ts)
        gaps = [sorted_ts[i+1] - sorted_ts[i] for i in range(len(sorted_ts)-1)]
        print(f'  Gap stats (seconds):   min={min(gaps):.3f}  median={np.median(gaps):.3f}  '
              f'max={max(gaps):.1f}  p95={np.percentile(gaps,95):.2f}')
        gaps_above = {g: sum(1 for x in gaps if x > g) for g in [0.5, 1.0, 2.0, 3.0, 5.0]}
        print(f'  Gaps > threshold:      ' +
              '  '.join(f'{g}s→{c}' for g, c in gaps_above.items()))

    exif_usable = ts_available >= len(raw_paths) * 0.5 and ts_unique > 1
    print(f'  EXIF usable for burst grouping: {exif_usable}')

    # ── Step 3: Sort filenames ────────────────────────────────────────────────
    fnames_sorted = sorted(ts_by_fname.keys(), key=file_num)

    # ── Step 4: EXIF burst sweep ──────────────────────────────────────────────
    print('\nStep 3: EXIF burst grouping sweep')
    print(f'  {"gap_thresh":>10} {"cluster_t":>10} {"n_bursts":>9} {"n_clusters":>11} '
          f'{"coverage":>9} {"purity":>7} {"composite":>10}')

    gap_thresholds   = [0.5, 1.0, 2.0, 3.0, 5.0]
    cluster_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

    best_exif = {'composite': -1.0}
    exif_results: list[dict] = []

    for gap_thresh in gap_thresholds:
        if exif_usable:
            bursts = group_by_exif_gap(fnames_sorted, ts_by_fname, gap_thresh)
        else:
            # Fallback: use fixed window equal to approximate burst size implied by gap
            approx_k = max(3, int(gap_thresh * 5))
            bursts = group_by_fixed_window(fnames_sorted, approx_k)
            print(f'  (EXIF unavailable — using fixed window K={approx_k} for gap={gap_thresh}s)')

        valid_bursts, burst_matrix = burst_embeddings(bursts, emb_by_fname)

        for ct in cluster_thresholds:
            m = compute_metrics(valid_bursts, burst_matrix, ct, true_player_fn, n_true_players)
            m['gap_thresh'] = gap_thresh
            m['cluster_thresh'] = ct
            m['method'] = 'exif_burst' if exif_usable else 'fallback_window'
            exif_results.append(m)

            tag = 'OK' if m['coverage_count'] == n_true_players else f'miss{n_true_players - m["coverage_count"]}'
            print(f'  gap={gap_thresh:4.1f}s  ct={ct:.2f}  bursts={m["n_bursts"]:5d}  '
                  f'clusters={m["n_clusters"]:4d}  '
                  f'cov={m["coverage_count"]}/{n_true_players}  '
                  f'purity={m["purity"]:.3f}  comp={m["composite"]:.3f}  {tag}')

            if m['composite'] > best_exif['composite']:
                best_exif = dict(m)
        print()

    # ── Step 5: Fixed-window fallback sweep ───────────────────────────────────
    print('\nStep 4: Fixed-window fallback sweep (K = window size)')
    print(f'  {"K":>4} {"cluster_t":>10} {"n_bursts":>9} {"n_clusters":>11} '
          f'{"coverage":>9} {"purity":>7} {"composite":>10}')

    window_sizes = [3, 5, 7, 10]
    best_window = {'composite': -1.0}
    window_results: list[dict] = []

    for K in window_sizes:
        bursts = group_by_fixed_window(fnames_sorted, K)
        valid_bursts, burst_matrix = burst_embeddings(bursts, emb_by_fname)

        for ct in cluster_thresholds:
            m = compute_metrics(valid_bursts, burst_matrix, ct, true_player_fn, n_true_players)
            m['K'] = K
            m['cluster_thresh'] = ct
            m['method'] = 'fixed_window'
            window_results.append(m)

            tag = 'OK' if m['coverage_count'] == n_true_players else f'miss{n_true_players - m["coverage_count"]}'
            print(f'  K={K:2d}  ct={ct:.2f}  bursts={m["n_bursts"]:5d}  '
                  f'clusters={m["n_clusters"]:4d}  '
                  f'cov={m["coverage_count"]}/{n_true_players}  '
                  f'purity={m["purity"]:.3f}  comp={m["composite"]:.3f}  {tag}')

            if m['composite'] > best_window['composite']:
                best_window = dict(m)
        print()

    # ── Step 6: No-grouping baseline ─────────────────────────────────────────
    print('\nStep 5: No-grouping baseline (each photo = its own burst)')
    bursts_base = group_no_grouping(fnames_sorted)
    valid_bursts_base, burst_matrix_base = burst_embeddings(bursts_base, emb_by_fname)

    best_base = {'composite': -1.0}
    base_results: list[dict] = []

    for ct in cluster_thresholds:
        m = compute_metrics(valid_bursts_base, burst_matrix_base, ct, true_player_fn, n_true_players)
        m['cluster_thresh'] = ct
        m['method'] = 'no_grouping'
        base_results.append(m)
        tag = 'OK' if m['coverage_count'] == n_true_players else f'miss{n_true_players - m["coverage_count"]}'
        print(f'  ct={ct:.2f}  clusters={m["n_clusters"]:4d}  '
              f'cov={m["coverage_count"]}/{n_true_players}  '
              f'purity={m["purity"]:.3f}  comp={m["composite"]:.3f}  {tag}')
        if m['composite'] > best_base['composite']:
            best_base = dict(m)

    # ── Step 7: Comparison summary ────────────────────────────────────────────
    print(f'\n{"-"*70}')
    print('COMPARISON SUMMARY')
    print(f'{"-"*70}')
    method_label = 'EXIF burst' if exif_usable else 'Fallback window (EXIF unavail)'
    print(f'  {method_label:40s}  comp={best_exif["composite"]:.3f}  '
          f'cov={best_exif["coverage_count"]}/{n_true_players}  '
          f'purity={best_exif["purity"]:.3f}  '
          f'[gap={best_exif.get("gap_thresh","?")} ct={best_exif["cluster_thresh"]}]')
    print(f'  {"Fixed-window fallback":40s}  comp={best_window["composite"]:.3f}  '
          f'cov={best_window["coverage_count"]}/{n_true_players}  '
          f'purity={best_window["purity"]:.3f}  '
          f'[K={best_window.get("K","?")} ct={best_window["cluster_thresh"]}]')
    print(f'  {"No-grouping baseline":40s}  comp={best_base["composite"]:.3f}  '
          f'cov={best_base["coverage_count"]}/{n_true_players}  '
          f'purity={best_base["purity"]:.3f}  '
          f'[ct={best_base["cluster_thresh"]}]')

    # ── Step 8: 20%-kept culling simulation ───────────────────────────────────
    print(f'\n{"-"*70}')
    print('SIMULATION: 20% keep rate (random culling per player)')
    print(f'{"-"*70}')

    culled_fnames = simulate_culled(fnames_sorted, true_player_fn, n_true_players)
    print(f'  Photos after culling: {len(culled_fnames)} / {len(fnames_sorted)}')

    # Re-run best EXIF and best window configs on culled set
    for label, method_fn, best_cfg in [
        ('EXIF burst (best config)' if exif_usable else 'Fallback window (best config)',
         lambda fn: (
             group_by_exif_gap(fn, ts_by_fname, best_exif.get('gap_thresh', 2.0))
             if exif_usable else
             group_by_fixed_window(fn, best_exif.get('K', 5))
         ),
         best_exif),
        ('Fixed-window (best config)',
         lambda fn: group_by_fixed_window(fn, best_window.get('K', 5)),
         best_window),
        ('No-grouping baseline',
         lambda fn: group_no_grouping(fn),
         best_base),
    ]:
        culled_sorted = [f for f in fnames_sorted if f in set(culled_fnames)]
        bursts_c = method_fn(culled_sorted)
        vb_c, bm_c = burst_embeddings(bursts_c, emb_by_fname)
        ct = best_cfg.get('cluster_thresh', 0.65)
        m_c = compute_metrics(vb_c, bm_c, ct, true_player_fn, n_true_players)
        tag = 'OK' if m_c['coverage_count'] == n_true_players else f'miss {n_true_players - m_c["coverage_count"]}'
        print(f'  {label:45s}  cov={m_c["coverage_count"]}/{n_true_players}  '
              f'purity={m_c["purity"]:.3f}  comp={m_c["composite"]:.3f}  {tag}')

    print(f'\n{"="*70}')
    print(f'END: {name}')
    print(f'{"="*70}\n')


def main():
    print('=' * 70)
    print('EXIF Burst-Level Player Identification Evaluation')
    print('=' * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\nLoading models...')
    detector = build_detector()
    embedder = build_embedder(device)
    print('Models loaded.')

    # ── Primary: WA Open ──────────────────────────────────────────────────────
    wa_dir = Path(r'C:\Users\erikg\Downloads\Jay Ma Photography Training\202605 WA Open MS Raws')
    _seen: set[str] = set()
    wa_paths: list[Path] = []
    for p in sorted(wa_dir.glob('*'), key=lambda x: file_num(x.name)):
        if p.suffix.lower() != '.cr3' or p.name.lower() in _seen:
            continue
        _seen.add(p.name.lower())
        n = file_num(p.name)
        if 1 <= n <= 696:
            wa_paths.append(p)

    run_dataset(
        name='WA Open MS (202605) — 0001-0696',
        raw_dir=wa_dir,
        raw_paths=wa_paths,
        true_player_fn=wa_true_player,
        n_true_players=WA_N_PLAYERS,
        detector=detector,
        embedder=embedder,
        device=device,
    )

    # ── Secondary: Demo ───────────────────────────────────────────────────────
    demo_dir = Path(r'C:\Users\erikg\Downloads\Demo')
    if demo_dir.exists():
        demo_paths: list[Path] = []
        _seen2: set[str] = set()
        for p in sorted(demo_dir.glob('*'), key=lambda x: file_num(x.name)):
            if p.suffix.lower() == '.cr3' and p.name.lower() not in _seen2:
                _seen2.add(p.name.lower())
                demo_paths.append(p)

        run_dataset(
            name='Demo — 5 players',
            raw_dir=demo_dir,
            raw_paths=demo_paths,
            true_player_fn=demo_true_player,
            n_true_players=DEMO_N_PLAYERS,
            detector=detector,
            embedder=embedder,
            device=device,
        )
    else:
        print(f'\nDemo directory not found at {demo_dir} — skipping.')


if __name__ == '__main__':
    main()

"""
Test temporal changepoint clustering on WA Open dataset.
Approach: sort photos by filename, detect player changes when
consecutive embedding distance > change_threshold, average within segment,
cluster segment embeddings.
"""
import re
from pathlib import Path
import numpy as np
import torch
from inference.player_coverage import build_detector, build_embedder, detect_player_crops, embed_crops, cluster_embeddings
from data.raw_reader import extract_thumbnail

RANGES = [
    (1,    78,   1), (79,   107,  2), (108,  126,  3), (127,  150,  2),
    (151,  177,  4), (178,  186,  5), (187,  194,  6), (195,  227,  5),
    (228,  252,  6), (253,  290,  7), (291,  296,  8), (297,  327,  9),
    (328,  347,  10), (348,  401,  11), (402,  442,  12), (443,  473,  13),
    (474,  485,  14), (486,  509,  15), (510,  527,  14), (528,  575,  16),
    (576,  595,  17), (596,  634,  18), (635,  653,  19), (654,  696,  20),
]
def file_num(fname): m = re.search(r'(\d{4,})', Path(fname).stem); return int(m.group(1)) if m else -1
def true_player(fname):
    n = file_num(fname)
    for s,e,pid in RANGES:
        if s<=n<=e: return pid
    return 0

raw_dir = Path(r'C:\Users\erikg\Downloads\Jay Ma Photography Training\202605 WA Open MS Raws')
_seen = set()
labeled_raws = []
for p in sorted(raw_dir.glob('*'), key=lambda x: file_num(x.name)):
    if p.suffix.lower() != '.cr3' or p.name.lower() in _seen: continue
    _seen.add(p.name.lower())
    if 1 <= file_num(p.name) <= 696: labeled_raws.append(p)
print(f'Photos: {len(labeled_raws)}  True players: 20')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = build_detector()
embedder = build_embedder(device)

# One embedding per photo: largest detected person
print('Embedding...')
photo_embs, photo_names = [], []
for p in labeled_raws:
    img = extract_thumbnail(p, size=(448, 448))
    results = detector(img, classes=[0], verbose=False)
    w, h = img.size
    min_area = w * h * 0.01
    boxes = [(int(b[0]),int(b[1]),int(b[2]),int(b[3])) for r in results
             for b in (r.boxes.xyxy.cpu().tolist(),)[0]
             if (b[2]-b[0])*(b[3]-b[1]) >= min_area]
    if not boxes: continue
    largest = max(boxes, key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
    emb = embed_crops([img.crop(largest)], embedder, device)[0]
    photo_embs.append(emb)
    photo_names.append(p.name)
print(f'Detected: {len(photo_embs)}/{len(labeled_raws)}')


def temporal_cluster(photo_names, photo_embs, change_thresh, cluster_thresh):
    """
    1. Detect changepoints (consecutive cosine dist > change_thresh) → segments
    2. Average embeddings within each segment
    3. Cluster segment embeddings with cluster_thresh
    Returns (n_clusters, covered_players)
    """
    # Sort by file number (already sorted, but ensure it)
    order = sorted(range(len(photo_names)), key=lambda i: file_num(photo_names[i]))
    names = [photo_names[i] for i in order]
    embs  = [photo_embs[i]  for i in order]

    segments = []   # list of (list_of_names, avg_embedding)
    seg_names = [names[0]]
    seg_embs  = [embs[0]]
    seg_avg   = embs[0] / (np.linalg.norm(embs[0]) + 1e-8)

    for i in range(1, len(names)):
        e = embs[i] / (np.linalg.norm(embs[i]) + 1e-8)
        dist = 1.0 - float(np.dot(seg_avg, e))
        if dist > change_thresh:
            avg = np.mean(seg_embs, axis=0); avg /= np.linalg.norm(avg)+1e-8
            segments.append((seg_names, avg))
            seg_names, seg_embs, seg_avg = [names[i]], [embs[i]], e.copy()
        else:
            seg_names.append(names[i]); seg_embs.append(embs[i])
            seg_avg = np.mean(seg_embs, axis=0); seg_avg /= np.linalg.norm(seg_avg)+1e-8
    avg = np.mean(seg_embs, axis=0); avg /= np.linalg.norm(avg)+1e-8
    segments.append((seg_names, avg))

    seg_matrix = np.stack([s[1] for s in segments])
    if len(segments) == 1:
        labels = np.array([0])
    else:
        labels = cluster_embeddings(seg_matrix, cluster_thresh)
    n_clusters = int(labels.max()) + 1

    # Check which true players are covered
    covered = set()
    for seg_idx, (seg_n, _) in enumerate(segments):
        cid = int(labels[seg_idx])
        for fname in seg_n:
            covered.add(true_player(fname))
    covered.discard(0)
    return n_clusters, len(segments), covered


print('\n--- Baseline (no temporal) ---')
emb_matrix = np.stack(photo_embs)
for t in [0.50, 0.60, 0.65, 0.70]:
    labels = cluster_embeddings(emb_matrix, t)
    n = int(labels.max())+1
    covered = set()
    for i,fn in enumerate(photo_names): covered.add(true_player(fn))
    covered.discard(0)
    print(f'  thresh={t:.2f}: {n:3d} clusters  covered={len(covered)}/20')

print('\n--- Temporal changepoint clustering ---')
print(f'{"change_t":>10} {"cluster_t":>10} {"n_segs":>8} {"n_clusters":>11} {"covered":>8}')
for ct in [0.15, 0.20, 0.25, 0.30]:
    for ht in [0.50, 0.60, 0.65, 0.70]:
        n_cl, n_seg, covered = temporal_cluster(photo_names, photo_embs, ct, ht)
        ok = 'OK' if len(covered)==20 else f'miss {20-len(covered)}'
        print(f'  ct={ct:.2f}  ht={ht:.2f}  segs={n_seg:4d}  clusters={n_cl:3d}  {len(covered)}/20  {ok}')
    print()

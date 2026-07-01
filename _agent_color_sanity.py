"""
Detailed sanity check for Approach A (alpha=0.3, t=0.50) — the configuration
with the best ARI (0.428). Shows per-cluster player composition and
per-player cluster assignment.
"""
from __future__ import annotations
import re, sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import cv2, torch

sys.path.insert(0, r'C:\Users\erikg\OneDrive\Desktop\CodePlayground\badminton-photo-editor')
from inference.player_coverage import build_detector, build_embedder, embed_crops, cluster_embeddings
from data.raw_reader import extract_thumbnail

RANGES = [
    (1,78,1),(79,107,2),(108,126,3),(127,150,2),(151,177,4),(178,186,5),(187,194,6),(195,227,5),
    (228,252,6),(253,290,7),(291,296,8),(297,327,9),(328,347,10),(348,401,11),(402,442,12),
    (443,473,13),(474,485,14),(486,509,15),(510,527,14),(528,575,16),(576,595,17),(596,634,18),
    (635,653,19),(654,696,20),
]
def file_num(f): m=re.search(r'(\d{4,})',Path(f).stem); return int(m.group(1)) if m else -1
def true_player(f):
    n=file_num(f)
    for s,e,p in RANGES:
        if s<=n<=e: return p
    return 0

raw_dir = Path(r'C:\Users\erikg\Downloads\Jay Ma Photography Training\202605 WA Open MS Raws')
labeled_raws = sorted([p for p in raw_dir.glob('*.CR3') if 1<=file_num(p.name)<=696],
                      key=lambda x: file_num(x.name))
print(f'Photos: {len(labeled_raws)}')

import pickle
cache_path = Path('_agent_color_cache.pkl')

if cache_path.exists():
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    photo_names = data['names']
    dino_matrix = data['dino']
    hist_matrix = data['hist']
    print(f'Loaded cache: {len(photo_names)} photos')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = build_detector()
    embedder = build_embedder(device)

    def hsv_histogram(crop_rgb, n_bins=32):
        w, h = crop_rgb.size
        upper = crop_rgb.crop((0, 0, w, int(h*0.60)))
        arr = np.array(upper, dtype=np.uint8)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        hists = []
        ranges_ch = [(0,180),(0,256),(0,256)]
        for ch in range(3):
            hv = cv2.calcHist([hsv],[ch],None,[n_bins],[ranges_ch[ch][0],ranges_ch[ch][1]])
            hv = hv.flatten().astype(np.float32)
            t = hv.sum(); hv = hv/t if t>0 else hv
            hists.append(hv)
        return np.concatenate(hists)

    photo_names, dino_embs, color_hists = [], [], []
    for i, p in enumerate(labeled_raws):
        img = extract_thumbnail(p, size=(448,448))
        results = detector(img, classes=[0], verbose=False)
        w, h = img.size; min_area = w*h*0.01
        boxes = [(int(b[0]),int(b[1]),int(b[2]),int(b[3]))
                 for r in results for b in (r.boxes.xyxy.cpu().tolist(),)[0]
                 if (b[2]-b[0])*(b[3]-b[1]) >= min_area]
        if not boxes: continue
        largest = max(boxes, key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
        crop = img.crop(largest)
        try:
            emb = embed_crops([crop], embedder, device)[0]
            hist = hsv_histogram(crop)
        except: continue
        photo_names.append(p.name)
        dino_embs.append(emb)
        color_hists.append(hist)
        if (i+1)%100==0: print(f'  {i+1}/{len(labeled_raws)}')

    dino_matrix = np.stack(dino_embs)
    hist_matrix = np.stack(color_hists)
    with open(cache_path, 'wb') as f:
        pickle.dump({'names':photo_names,'dino':dino_matrix,'hist':hist_matrix}, f)
    print(f'Cached {len(photo_names)} photos')

# ── PCA-reduce histogram ──────────────────────────────────────────────────────
scaler = StandardScaler()
hist_scaled = scaler.fit_transform(hist_matrix)
pca = PCA(n_components=20, random_state=42)
hist_pca = pca.fit_transform(hist_scaled)
hnorms = np.linalg.norm(hist_pca, axis=1, keepdims=True)
hist_pca_norm = hist_pca / (hnorms + 1e-8)

# ── Approach A: alpha=0.3, t=0.50 ────────────────────────────────────────────
print('\n' + '='*70)
print('Approach A: alpha=0.3, thresh=0.50 — Detailed cluster breakdown')
print('='*70)

alpha = 0.3
fused = np.concatenate([alpha*dino_matrix, (1-alpha)*hist_pca_norm], axis=1)
fnorms = np.linalg.norm(fused, axis=1, keepdims=True)
fused_norm = fused / (fnorms + 1e-8)
labels = cluster_embeddings(fused_norm, 0.50)

cluster_to_players: dict[int,dict[int,int]] = {}
player_to_clusters: dict[int,set] = {}
for i, fname in enumerate(photo_names):
    pid = true_player(fname)
    cid = int(labels[i])
    cluster_to_players.setdefault(cid,{}).setdefault(pid,0)
    cluster_to_players[cid][pid] += 1
    if pid > 0:
        player_to_clusters.setdefault(pid,set()).add(cid)

sizes = {c:sum(v.values()) for c,v in cluster_to_players.items()}
n_clusters = len(cluster_to_players)
n_pure = sum(1 for ps in cluster_to_players.values() if len(ps)==1)

print(f'Total clusters: {n_clusters}  Pure: {n_pure}  Mixed: {n_clusters-n_pure}')
print()
print(f'{"Cluster":>8} {"Size":>6}  Players')
print('-'*70)
for cid in sorted(cluster_to_players, key=lambda c:-sizes[c]):
    ps = cluster_to_players[cid]
    sz = sizes[cid]
    label = 'PURE' if len(ps)==1 else f'MIX({len(ps)})'
    pstr = '  '.join(f'P{p}:{cnt}' for p,cnt in sorted(ps.items(),key=lambda x:-x[1]))
    print(f'  C{cid:04d}  {sz:5d}  [{label}]  {pstr}')

print('\n--- Player split analysis ---')
print('(Shows which players are cleanly in one cluster vs split across many)')
for pid in range(1,21):
    clusters = sorted(player_to_clusters.get(pid, set()))
    if len(clusters) <= 1:
        status = 'OK (1 cluster)'
    else:
        status = f'SPLIT across {len(clusters)} clusters: {clusters}'
    print(f'  P{pid:02d}: {status}')

# ── Color discriminability test ───────────────────────────────────────────────
print('\n' + '='*70)
print('COLOR DISCRIMINABILITY: HSV histogram inter-player distances')
print('='*70)
print('(Mean within-player hist distance vs between-player hist distance)')

# For each player, average their histograms
player_avg_hist: dict[int,np.ndarray] = {}
for i, fname in enumerate(photo_names):
    pid = true_player(fname)
    if pid == 0: continue
    player_avg_hist.setdefault(pid, []).append(hist_matrix[i])

player_mean_hist: dict[int,np.ndarray] = {}
for pid, hs in player_avg_hist.items():
    player_mean_hist[pid] = np.mean(hs, axis=0)

# Within-player variance
within_dists = []
for pid, hs in player_avg_hist.items():
    mean = player_mean_hist[pid]
    for h in hs:
        d = np.sqrt(np.sum((h-mean)**2))
        within_dists.append(d)

# Between-player distances
between_dists = []
pids = sorted(player_mean_hist.keys())
for i in range(len(pids)):
    for j in range(i+1, len(pids)):
        d = np.sqrt(np.sum((player_mean_hist[pids[i]]-player_mean_hist[pids[j]])**2))
        between_dists.append(d)

print(f'Within-player  histogram dist: mean={np.mean(within_dists):.4f} std={np.std(within_dists):.4f}')
print(f'Between-player histogram dist: mean={np.mean(between_dists):.4f} std={np.std(between_dists):.4f}')
ratio = np.mean(between_dists) / (np.mean(within_dists) + 1e-8)
print(f'Discrimination ratio (between/within): {ratio:.2f}  (>2 is useful, >5 is strong)')

print('\nPer-player dominant color (mean H, S, V in top HSV bins):')
for pid in sorted(player_mean_hist.keys()):
    h = player_mean_hist[pid]  # shape (96,)
    # H channel: first 32 bins, 0-180 range
    h_bins = h[:32]
    s_bins = h[32:64]
    v_bins = h[64:96]
    dom_h = np.argmax(h_bins) * (180/32)
    dom_s = np.argmax(s_bins) * (255/32)
    dom_v = np.argmax(v_bins) * (255/32)
    print(f'  P{pid:02d}: dom_H={dom_h:5.1f}  dom_S={dom_s:5.1f}  dom_V={dom_v:5.1f}')

print('\n[Done]')

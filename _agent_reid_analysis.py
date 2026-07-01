"""
Deep analysis of the PCA-whitened DINOv2 approach.

Key finding from _agent_reid.py:
- C2 (PCA whitened) achieves 0.997 composite but 377 clusters for 20 players
- This means it correctly splits photos into ~19x too many clusters
- BUT coverage is 20/20 — all players found
- ARI is only 0.063 — cluster assignments are noisy

This analysis:
1. Checks what ARI looks like at different cluster thresholds for whitened embeddings
2. Compares whitened vs baseline at different thresholds to understand the tradeoff
3. Investigates OSNet ARI=0.166 (much better ARI) — why does it score best on ARI?
4. Tests temporal smoothing on OSNet embeddings (since OSNet has best ARI)
5. Tests whitened DINOv2 + temporal aggregation
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

sys.path.insert(0, r'C:\Users\erikg\OneDrive\Desktop\CodePlayground\badminton-photo-editor')
from inference.player_coverage import build_detector, embed_crops, cluster_embeddings
from data.raw_reader import extract_thumbnail

# Ground truth
RANGES = [
    (1,78,1),(79,107,2),(108,126,3),(127,150,2),
    (151,177,4),(178,186,5),(187,194,6),(195,227,5),
    (228,252,6),(253,290,7),(291,296,8),(297,327,9),
    (328,347,10),(348,401,11),(402,442,12),(443,473,13),
    (474,485,14),(486,509,15),(510,527,14),(528,575,16),
    (576,595,17),(596,634,18),(635,653,19),(654,696,20),
]

def file_num(fname):
    m = re.search(r'(\d{4,})', Path(fname).stem)
    return int(m.group(1)) if m else -1

def true_player(fname):
    n = file_num(fname)
    for s, e, pid in RANGES:
        if s <= n <= e: return pid
    return 0

_REID_TF = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def pca_whiten(embs, n_components=64):
    mean = embs.mean(axis=0, keepdims=True)
    centered = embs - mean
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, whiten=True)
    whitened = pca.fit_transform(centered)
    norms = np.linalg.norm(whitened, axis=1, keepdims=True) + 1e-8
    return whitened / norms

def eval_thresh(embs, fnames, thresh, n_true=20):
    labels = cluster_embeddings(embs, thresh)
    n_cl = int(labels.max()) + 1
    p2c, c2p = {}, {}
    gt, pred = [], []
    for i, fn in enumerate(fnames):
        pid = true_player(fn)
        cid = int(labels[i])
        gt.append(pid)
        pred.append(cid)
        p2c.setdefault(pid, set()).add(cid)
        c2p.setdefault(cid, set()).add(pid)
    covered = set(p2c.keys()) - {0}
    pure = sum(1 for ps in c2p.values() if len(ps)==1)
    purity = pure / n_cl
    cov_frac = len(covered) / n_true
    comp = cov_frac * purity
    # ARI on labeled
    lm = [pid != 0 for pid in gt]
    gt_arr = np.array([gt[i] for i, m in enumerate(lm) if m])
    pred_arr = np.array([pred[i] for i, m in enumerate(lm) if m])
    ari = adjusted_rand_score(gt_arr, pred_arr)
    return {'n_cl': n_cl, 'covered': len(covered), 'pure': pure, 'purity': purity,
            'comp': comp, 'ari': ari, 'cov': cov_frac}


def temporal_segment_embed(photo_names, photo_embs, change_thresh):
    """Return (segment_names_list, segment_avg_embs) using temporal changepoint."""
    order = sorted(range(len(photo_names)), key=lambda i: file_num(photo_names[i]))
    names = [photo_names[i] for i in order]
    embs = [photo_embs[i] for i in order]

    segments = []
    seg_names = [names[0]]
    seg_embs = [embs[0]]
    seg_avg = embs[0] / (np.linalg.norm(embs[0]) + 1e-8)

    for i in range(1, len(names)):
        e = embs[i] / (np.linalg.norm(embs[i]) + 1e-8)
        dist = 1.0 - float(np.dot(seg_avg, e))
        if dist > change_thresh:
            avg = np.mean(seg_embs, axis=0); avg /= np.linalg.norm(avg) + 1e-8
            segments.append((seg_names, avg))
            seg_names, seg_embs, seg_avg = [names[i]], [embs[i]], e.copy()
        else:
            seg_names.append(names[i]); seg_embs.append(embs[i])
            seg_avg = np.mean(seg_embs, axis=0); seg_avg /= np.linalg.norm(seg_avg) + 1e-8

    avg = np.mean(seg_embs, axis=0); avg /= np.linalg.norm(avg) + 1e-8
    segments.append((seg_names, avg))
    return segments


def eval_temporal(photo_names, photo_embs, change_thresh, cluster_thresh, n_true=20):
    segments = temporal_segment_embed(photo_names, photo_embs, change_thresh)
    seg_matrix = np.stack([s[1] for s in segments])
    if len(segments) == 1:
        seg_labels = np.array([0])
    else:
        seg_labels = cluster_embeddings(seg_matrix, cluster_thresh)

    n_cl = int(seg_labels.max()) + 1
    n_seg = len(segments)

    # Assign each photo to its segment's cluster
    fname_to_cluster = {}
    for si, (seg_ns, _) in enumerate(segments):
        cl = int(seg_labels[si])
        for fn in seg_ns:
            fname_to_cluster[fn] = cl

    p2c, c2p = {}, {}
    gt_labels, pred_labels = [], []
    for fn in photo_names:
        pid = true_player(fn)
        cid = fname_to_cluster.get(fn, 0)
        gt_labels.append(pid)
        pred_labels.append(cid)
        p2c.setdefault(pid, set()).add(cid)
        c2p.setdefault(cid, set()).add(pid)

    covered = set(p2c.keys()) - {0}
    pure = sum(1 for ps in c2p.values() if len(ps)==1)
    purity = pure / n_cl
    cov_frac = len(covered) / n_true
    comp = cov_frac * purity

    lm = [pid != 0 for pid in gt_labels]
    gt_arr = np.array([gt_labels[i] for i, m in enumerate(lm) if m])
    pred_arr = np.array([pred_labels[i] for i, m in enumerate(lm) if m])
    ari = adjusted_rand_score(gt_arr, pred_arr)
    return {'n_cl': n_cl, 'n_seg': n_seg, 'covered': len(covered), 'purity': purity,
            'comp': comp, 'ari': ari, 'cov': cov_frac}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    wa_dir = Path(r'C:\Users\erikg\Downloads\Jay Ma Photography Training\202605 WA Open MS Raws')
    _seen = set()
    wa_raws = []
    for p in sorted(wa_dir.glob('*.CR3'), key=lambda x: x.name):
        if p.name.lower() in _seen: continue
        _seen.add(p.name.lower())
        n = file_num(p.name)
        if 1 <= n <= 696: wa_raws.append(p)
    print(f"WA Open: {len(wa_raws)} photos")

    # Build models
    detector = build_detector()
    from inference.player_coverage import build_embedder
    dinov2 = build_embedder(device)

    import torchreid
    osnet_model = torchreid.models.build_model(name='osnet_x1_0', num_classes=1000, pretrained=True)
    osnet_model.classifier = torch.nn.Identity()
    osnet_model = osnet_model.eval().to(device)

    # Detect + embed
    print("\nDetecting + embedding...")
    wa_paths, wa_crops, wa_fnames = [], [], []
    for i, p in enumerate(wa_raws):
        try:
            img = extract_thumbnail(p, size=(448, 448))
            results = detector(img, classes=[0], verbose=False)
            w, h = img.size
            min_area = w * h * 0.01
            boxes = []
            for r in results:
                for box in r.boxes.xyxy.cpu().tolist():
                    x1, y1, x2, y2 = map(int, box)
                    if (x2-x1)*(y2-y1) >= min_area:
                        boxes.append((x1,y1,x2,y2))
            if not boxes: continue
            largest = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            crop = img.crop(largest)
            wa_paths.append(p); wa_crops.append(crop); wa_fnames.append(p.name)
        except: pass
        if (i+1)%100==0: print(f"  {i+1}/{len(wa_raws)} ({len(wa_fnames)} detected)")
    print(f"Detected: {len(wa_fnames)}/{len(wa_raws)}")

    # Embeddings
    print("Computing DINOv2 embeddings...")
    wa_dinov2 = np.stack([embed_crops([c], dinov2, device)[0] for c in wa_crops])

    print("Computing OSNet embeddings...")
    wa_osnet_embs = []
    for c in wa_crops:
        batch = _REID_TF(c).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = osnet_model(batch)
            feat = F.normalize(feat, dim=1)
        wa_osnet_embs.append(feat.cpu().float().numpy()[0])
    wa_osnet = np.stack(wa_osnet_embs)

    # === Analysis 1: Threshold sweep comparison ===
    print("\n" + "="*70)
    print("=== THRESHOLD SWEEP: DINOv2 vs Whitened DINOv2 vs OSNet ===")
    print("="*70)
    print(f"\n{'Thresh':>8}  {'DINOv2':^30}  {'Whitened':^30}  {'OSNet':^30}")
    print(f"{'':8}  {'n_cl Cov  Pur  Comp ARI':^30}  {'n_cl Cov  Pur  Comp ARI':^30}  {'n_cl Cov  Pur  Comp ARI':^30}")
    print("-"*100)

    for n_comp in [32, 64, 128]:
        wa_whitened = pca_whiten(wa_dinov2, n_components=n_comp)
        print(f"\n[PCA n_components={n_comp}]")
        for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            d = eval_thresh(wa_dinov2, wa_fnames, thresh)
            w = eval_thresh(wa_whitened, wa_fnames, thresh)
            o = eval_thresh(wa_osnet, wa_fnames, thresh)
            print(f"  t={thresh:.2f}  "
                  f"DINOv2: {d['n_cl']:3d} cl {d['covered']:2d}/20 pur={d['purity']:.3f} comp={d['comp']:.3f} ARI={d['ari']:.3f}  "
                  f"White: {w['n_cl']:3d} cl {w['covered']:2d}/20 pur={w['purity']:.3f} comp={w['comp']:.3f} ARI={w['ari']:.3f}  "
                  f"OSNet: {o['n_cl']:3d} cl {o['covered']:2d}/20 pur={o['purity']:.3f} comp={o['comp']:.3f} ARI={o['ari']:.3f}")

    # === Analysis 2: OSNet + temporal segmentation ===
    print("\n" + "="*70)
    print("=== TEMPORAL SEGMENTATION ON OSNET EMBEDDINGS ===")
    print("="*70)
    print(f"\n{'ct':>6}  {'ht':>6}  {'segs':>6}  {'cl':>5}  {'cov':>5}  {'pur':>6}  {'comp':>7}  {'ARI':>7}")
    for ct in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        for ht in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            r = eval_temporal(wa_fnames, list(wa_osnet), ct, ht)
            print(f"  ct={ct:.2f}  ht={ht:.2f}  segs={r['n_seg']:4d}  cl={r['n_cl']:3d}  "
                  f"cov={r['covered']:2d}/20  pur={r['purity']:.3f}  comp={r['comp']:.3f}  ARI={r['ari']:.3f}")
        print()

    # === Analysis 3: PCA whitened + temporal ===
    print("\n" + "="*70)
    print("=== TEMPORAL SEGMENTATION ON PCA-WHITENED DINOV2 ===")
    print("="*70)
    wa_whitened_64 = pca_whiten(wa_dinov2, n_components=64)
    print(f"\n{'ct':>6}  {'ht':>6}  {'segs':>6}  {'cl':>5}  {'cov':>5}  {'pur':>6}  {'comp':>7}  {'ARI':>7}")
    for ct in [0.10, 0.20, 0.30, 0.40, 0.50]:
        for ht in [0.35, 0.45, 0.55, 0.65, 0.75]:
            r = eval_temporal(wa_fnames, list(wa_whitened_64), ct, ht)
            print(f"  ct={ct:.2f}  ht={ht:.2f}  segs={r['n_seg']:4d}  cl={r['n_cl']:3d}  "
                  f"cov={r['covered']:2d}/20  pur={r['purity']:.3f}  comp={r['comp']:.3f}  ARI={r['ari']:.3f}")
        print()

    # === Analysis 4: Hybrid — OSNet + whitened DINOv2 (parity blend) ===
    print("\n" + "="*70)
    print("=== HYBRID: OSNet + PCA-whitened DINOv2 with temporal ===")
    print("="*70)
    wa_whitened_64 = pca_whiten(wa_dinov2, n_components=64)
    # Try different alpha blend ratios
    for alpha in [0.3, 0.5, 0.7]:
        # Normalize each to unit length before blending
        osnet_n = wa_osnet / (np.linalg.norm(wa_osnet, axis=1, keepdims=True) + 1e-8)
        white_n = wa_whitened_64 / (np.linalg.norm(wa_whitened_64, axis=1, keepdims=True) + 1e-8)
        # Pad OSNet to same dim via PCA → just concatenate and re-normalize
        blended = np.concatenate([alpha * osnet_n, (1-alpha) * white_n], axis=1)
        blended /= np.linalg.norm(blended, axis=1, keepdims=True) + 1e-8
        print(f"\n  alpha={alpha:.1f} (OSNet weight={alpha}, Whitened DINOv2 weight={1-alpha:.1f}):")
        for ct in [0.15, 0.25, 0.35]:
            for ht in [0.35, 0.45, 0.55]:
                r = eval_temporal(wa_fnames, list(blended), ct, ht)
                print(f"    ct={ct:.2f}  ht={ht:.2f}  segs={r['n_seg']:4d}  cl={r['n_cl']:3d}  "
                      f"cov={r['covered']:2d}/20  pur={r['purity']:.3f}  comp={r['comp']:.3f}  ARI={r['ari']:.3f}")

    # === Analysis 5: Per-player embedding spread (DINOv2 vs OSNet) ===
    print("\n" + "="*70)
    print("=== PER-PLAYER SPREAD ANALYSIS ===")
    print("="*70)
    print("\nFor each player: intra-player cosine distance spread (std of pairwise dists)")
    print(f"{'Player':>8}  {'N':>5}  {'DINOv2 intra':>14}  {'OSNet intra':>12}  {'Range (files)':>15}")

    player_embs_d = {}
    player_embs_o = {}
    for i, fn in enumerate(wa_fnames):
        pid = true_player(fn)
        if pid == 0: continue
        player_embs_d.setdefault(pid, []).append(wa_dinov2[i])
        player_embs_o.setdefault(pid, []).append(wa_osnet[i])

    # Get file ranges per player
    player_ranges = {}
    for s, e, pid in RANGES:
        if pid not in player_ranges:
            player_ranges[pid] = (s, e)
        else:
            pr = player_ranges[pid]
            player_ranges[pid] = (min(pr[0], s), max(pr[1], e))

    for pid in sorted(player_embs_d.keys()):
        de = np.stack(player_embs_d[pid])
        oe = np.stack(player_embs_o[pid])
        n = len(de)

        if n >= 2:
            # Mean intra-player cosine distance
            d_sims = de @ de.T
            mask = np.triu(np.ones((n, n), bool), k=1)
            d_intra = np.mean(1 - d_sims[mask])
            o_sims = oe @ oe.T
            o_intra = np.mean(1 - o_sims[mask])
        else:
            d_intra = 0.0
            o_intra = 0.0

        rng = player_ranges.get(pid, (0, 0))
        print(f"  Player {pid:2d}: n={n:3d}  DINOv2={d_intra:.4f}  OSNet={o_intra:.4f}  files={rng[0]}-{rng[1]}")

    print("\nDone.")


if __name__ == '__main__':
    main()

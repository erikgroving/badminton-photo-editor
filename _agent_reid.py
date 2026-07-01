"""
Agent: Novel Re-ID approaches for badminton player identification.

Tests three complementary approaches:
  A. OSNet person re-ID model (torchreid) — trained on Market-1501 person re-ID dataset
  B. Multi-scale body part features — full body + upper + lower + jersey patch concatenated
  C. PCA whitening of DINOv2 embeddings — centers and decorrelates embedding space

Also evaluates fusion of A+B+C to see if combination beats any single method.

Evaluation metrics:
  - coverage: how many of 20 true players have at least one cluster (goal: 20/20)
  - purity: fraction of clusters that are single-player
  - ARI: adjusted Rand index vs ground truth
  - composite: coverage_frac * purity
  - embedding space: intra/inter player cosine distances and their ratio
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

# ── Ground truth ───────────────────────────────────────────────────────────────

RANGES = [
    (1,    78,   1), (79,   107,  2), (108,  126,  3), (127,  150,  2),
    (151,  177,  4), (178,  186,  5), (187,  194,  6), (195,  227,  5),
    (228,  252,  6), (253,  290,  7), (291,  296,  8), (297,  327,  9),
    (328,  347,  10), (348,  401,  11), (402,  442,  12), (443,  473,  13),
    (474,  485,  14), (486,  509,  15), (510,  527,  14), (528,  575,  16),
    (576,  595,  17), (596,  634,  18), (635,  653,  19), (654,  696,  20),
]

DEMO_GROUPS = {
    1: set(range(1316, 1344)),
    2: set(range(1344, 1390)),
    3: set(range(1390, 1447)),
    4: set(range(1447, 1480)),
    5: set(range(1480, 1487)),
}


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


# ── OSNet Re-ID embedder ───────────────────────────────────────────────────────

_REID_TF = transforms.Compose([
    transforms.Resize((256, 128)),   # OSNet expects tall narrow crops
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_osnet(device: torch.device) -> torch.nn.Module:
    """Build OSNet-x1_0 as a feature extractor (remove classifier head)."""
    import torchreid
    # Build with pretrained ImageNet weights — Market-1501 weights require download
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=1000,
        pretrained=True,  # ImageNet pretrained
    )
    # Remove classifier: OSNet uses 'classifier' attribute
    # We want the feature extractor output (512-d for osnet_x1_0)
    model.eval()

    # Wrap to extract features only (strip the FC classifier)
    class OSNetFeatureExtractor(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            # Remove classifier so forward returns features
            self.base.classifier = torch.nn.Identity()

        def forward(self, x):
            return self.base(x)

    wrapper = OSNetFeatureExtractor(model)
    return wrapper.eval().to(device)


def embed_osnet(crops: list[Image.Image], osnet, device: torch.device) -> np.ndarray:
    """Embed crops using OSNet re-ID model. Returns L2-normalized features."""
    if not crops:
        return np.zeros((0, 512), dtype=np.float32)
    batch = torch.stack([_REID_TF(c) for c in crops]).to(device)
    with torch.no_grad():
        feats = osnet(batch)
        feats = F.normalize(feats, dim=1)
    return feats.cpu().float().numpy()


# ── Multi-scale body part crops ────────────────────────────────────────────────

def make_multiscale_crops(body_crop: Image.Image) -> list[Image.Image]:
    """
    From a full-body detection crop, extract sub-regions:
      - full body (100%)
      - upper body (top 55%)
      - lower body (bottom 55%)
      - jersey / torso patch (middle 30%, vertically centered)

    Returns list of 4 PIL crops.
    """
    w, h = body_crop.size
    parts = [
        body_crop,                                           # full body
        body_crop.crop((0, 0, w, int(h * 0.55))),           # upper body
        body_crop.crop((0, int(h * 0.45), w, h)),           # lower body
        body_crop.crop((0, int(h * 0.30), w, int(h * 0.70))),  # jersey/torso
    ]
    return parts


def embed_multiscale(
    body_crop: Image.Image,
    embedder,
    device: torch.device,
) -> np.ndarray:
    """
    Embed each scale part with DINOv2 and concatenate.
    Returns a single 4*768 = 3072-d normalized vector.
    """
    parts = make_multiscale_crops(body_crop)
    part_embs = embed_crops(parts, embedder, device)  # (4, 768)
    # L2-normalize each part, then concatenate
    normed = []
    for e in part_embs:
        n = np.linalg.norm(e) + 1e-8
        normed.append(e / n)
    combined = np.concatenate(normed)  # (3072,)
    combined /= np.linalg.norm(combined) + 1e-8
    return combined


# ── PCA whitening ──────────────────────────────────────────────────────────────

def pca_whiten(embs: np.ndarray, n_components: int = 128) -> np.ndarray:
    """
    Center + PCA whiten the embedding matrix.
    Projects to n_components dimensions, then scales each component by 1/sqrt(eigenvalue).
    Returns L2-normalized whitened embeddings.
    """
    # Center
    mean = embs.mean(axis=0, keepdims=True)
    centered = embs - mean

    # PCA with whitening
    pca = PCA(n_components=n_components, whiten=True)
    whitened = pca.fit_transform(centered)

    # L2-normalize rows
    norms = np.linalg.norm(whitened, axis=1, keepdims=True) + 1e-8
    return whitened / norms


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def evaluate_clusters(
    labels: np.ndarray,
    fnames: list[str],
    true_player_fn,
    n_true: int,
    tag: str,
) -> dict:
    """Compute coverage, purity, ARI, composite."""
    player_to_clusters: dict = {}
    cluster_to_players: dict = {}
    gt_labels = []
    pred_labels = []

    for i, fname in enumerate(fnames):
        pid = true_player_fn(fname)
        cid = int(labels[i])
        gt_labels.append(pid)
        pred_labels.append(cid)
        player_to_clusters.setdefault(pid, set()).add(cid)
        cluster_to_players.setdefault(cid, set()).add(pid)

    n_clusters = int(labels.max()) + 1
    covered = set(player_to_clusters.keys()) - {0}
    pure = sum(1 for ps in cluster_to_players.values() if len(ps) == 1)
    purity = pure / n_clusters if n_clusters > 0 else 0.0
    coverage_frac = len(covered) / n_true
    composite = coverage_frac * purity

    # ARI (only on labeled photos)
    labeled_mask = [pid != 0 for pid in gt_labels]
    gt_arr = np.array([gt_labels[i] for i, m in enumerate(labeled_mask) if m])
    pred_arr = np.array([pred_labels[i] for i, m in enumerate(labeled_mask) if m])
    ari = adjusted_rand_score(gt_arr, pred_arr) if len(gt_arr) > 1 else 0.0

    print(
        f"  [{tag}] thresh=best: {n_clusters:3d} clusters  "
        f"covered={len(covered)}/{n_true}  pure={pure}/{n_clusters}  "
        f"purity={purity:.3f}  composite={composite:.3f}  ARI={ari:.3f}"
    )
    return {
        'n_clusters': n_clusters, 'covered': len(covered), 'n_true': n_true,
        'pure': pure, 'purity': purity, 'coverage_frac': coverage_frac,
        'composite': composite, 'ari': ari,
    }


def sweep_thresholds(
    emb_matrix: np.ndarray,
    fnames: list[str],
    true_player_fn,
    n_true: int,
    tag: str,
    thresholds: list[float] | None = None,
) -> tuple[float, dict]:
    """Sweep clustering thresholds, return best threshold + best metrics dict."""
    if thresholds is None:
        thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    best_composite = -1.0
    best_thresh = 0.65
    best_metrics = {}
    best_labels = None

    for thresh in thresholds:
        labels = cluster_embeddings(emb_matrix, thresh)
        n_clusters = int(labels.max()) + 1

        player_to_clusters: dict = {}
        cluster_to_players: dict = {}
        for i, fname in enumerate(fnames):
            pid = true_player_fn(fname)
            cid = int(labels[i])
            player_to_clusters.setdefault(pid, set()).add(cid)
            cluster_to_players.setdefault(cid, set()).add(pid)

        covered = set(player_to_clusters.keys()) - {0}
        pure = sum(1 for ps in cluster_to_players.values() if len(ps) == 1)
        purity = pure / n_clusters if n_clusters > 0 else 0.0
        coverage_frac = len(covered) / n_true
        composite = coverage_frac * purity

        if composite > best_composite:
            best_composite = composite
            best_thresh = thresh
            best_labels = labels.copy()

    # Evaluate best threshold
    best_metrics = evaluate_clusters(best_labels, fnames, true_player_fn, n_true, f"{tag} (thresh={best_thresh:.2f})")
    best_metrics['best_thresh'] = best_thresh
    return best_thresh, best_metrics


def compute_embedding_stats(
    emb_matrix: np.ndarray,
    fnames: list[str],
    true_player_fn,
    tag: str,
) -> dict:
    """
    Compute intra-player and inter-player average cosine distances.
    Lower intra/inter ratio = better separation.
    """
    # Group by player
    player_embs: dict[int, list] = {}
    for i, fname in enumerate(fnames):
        pid = true_player_fn(fname)
        if pid != 0:
            player_embs.setdefault(pid, []).append(emb_matrix[i])

    # Intra-player distances
    intra_dists = []
    for pid, embs in player_embs.items():
        if len(embs) < 2:
            continue
        em = np.stack(embs)
        # Cosine distance = 1 - cosine_similarity
        # em is already (N, D) — compute pairwise
        sims = em @ em.T  # (N, N)
        mask = np.triu(np.ones((len(em), len(em)), dtype=bool), k=1)
        intra_dists.extend((1.0 - sims[mask]).tolist())

    # Inter-player distances (sample pairs of players)
    inter_dists = []
    player_ids = sorted(player_embs.keys())
    for i in range(len(player_ids)):
        for j in range(i + 1, len(player_ids)):
            p1_embs = np.stack(player_embs[player_ids[i]])
            p2_embs = np.stack(player_embs[player_ids[j]])
            # Mean embedding per player for efficiency
            m1 = p1_embs.mean(axis=0)
            m1 /= np.linalg.norm(m1) + 1e-8
            m2 = p2_embs.mean(axis=0)
            m2 /= np.linalg.norm(m2) + 1e-8
            inter_dists.append(1.0 - float(np.dot(m1, m2)))

    avg_intra = np.mean(intra_dists) if intra_dists else 0.0
    avg_inter = np.mean(inter_dists) if inter_dists else 0.0
    ratio = avg_intra / (avg_inter + 1e-8)

    print(f"\n  [{tag}] Embedding space stats:")
    print(f"    avg intra-player cosine dist: {avg_intra:.4f}")
    print(f"    avg inter-player cosine dist: {avg_inter:.4f}")
    print(f"    intra/inter ratio:            {ratio:.4f}  (< 1.0 = separable)")

    # Most confused player pairs (smallest inter-player distance)
    pair_dists = []
    for i in range(len(player_ids)):
        for j in range(i + 1, len(player_ids)):
            p1_embs = np.stack(player_embs[player_ids[i]])
            p2_embs = np.stack(player_embs[player_ids[j]])
            m1 = p1_embs.mean(axis=0); m1 /= np.linalg.norm(m1) + 1e-8
            m2 = p2_embs.mean(axis=0); m2 /= np.linalg.norm(m2) + 1e-8
            pair_dists.append((1.0 - float(np.dot(m1, m2)), player_ids[i], player_ids[j]))
    pair_dists.sort()

    print(f"    5 most confused player pairs (lowest inter-player distance):")
    for dist, p1, p2 in pair_dists[:5]:
        print(f"      Players {p1:2d} & {p2:2d}: cosine dist = {dist:.4f}")

    return {
        'avg_intra': avg_intra, 'avg_inter': avg_inter, 'ratio': ratio,
        'pair_dists': pair_dists,
    }


# ── Detect + collect crops ──────────────────────────────────────────────────────

def collect_photo_data(
    raw_paths: list[Path],
    detector,
    thumb_size: tuple = (448, 448),
) -> tuple[list[Path], list[Image.Image]]:
    """Detect largest person per photo. Returns (paths_with_detections, crops)."""
    detected_paths = []
    crops = []
    print(f"Detecting persons in {len(raw_paths)} photos...")
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
            crop = img.crop(largest)
            detected_paths.append(p)
            crops.append(crop)
        except Exception as e:
            pass
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(raw_paths)} ({len(crops)} detected)")
    print(f"  Done: {len(crops)}/{len(raw_paths)} photos with detections")
    return detected_paths, crops


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 70)

    # ── Load datasets ──────────────────────────────────────────────────────────
    wa_dir = Path(r'C:\Users\erikg\Downloads\Jay Ma Photography Training\202605 WA Open MS Raws')
    demo_dir = Path(r'C:\Users\erikg\Downloads\Demo')

    # WA Open labeled subset (files 1-696)
    _seen: set = set()
    wa_raws = []
    for p in sorted(wa_dir.glob('*.CR3'), key=lambda x: x.name):
        if p.name.lower() in _seen:
            continue
        _seen.add(p.name.lower())
        n = file_num(p.name)
        if 1 <= n <= 696:
            wa_raws.append(p)

    print(f"\nWA Open labeled: {len(wa_raws)} photos  (20 true players)")

    # Demo dataset
    _seen2: set = set()
    demo_raws = []
    for p in sorted(demo_dir.glob('*.CR3'), key=lambda x: x.name):
        if p.name.lower() not in _seen2:
            _seen2.add(p.name.lower())
            demo_raws.append(p)
    print(f"Demo:            {len(demo_raws)} photos  (5 true players)")

    # ── Build models ───────────────────────────────────────────────────────────
    print("\nBuilding models...")
    detector = build_detector()

    from inference.player_coverage import build_embedder
    dinov2 = build_embedder(device)
    print("  DINOv2-base loaded")

    osnet = build_osnet(device)
    print("  OSNet-x1_0 loaded (ImageNet pretrained)")

    # ── Detect + crop ──────────────────────────────────────────────────────────
    print("\n--- Detecting largest person per photo ---")
    wa_paths, wa_crops = collect_photo_data(wa_raws, detector)
    wa_fnames = [p.name for p in wa_paths]

    demo_paths, demo_crops = collect_photo_data(demo_raws, detector)
    demo_fnames = [p.name for p in demo_paths]

    # ── Compute all embeddings ─────────────────────────────────────────────────
    print("\nComputing embeddings...")

    # A: OSNet Re-ID features
    print("  [A] OSNet re-ID embedding...")
    wa_osnet = np.stack([embed_osnet([c], osnet, device)[0] for c in wa_crops])
    demo_osnet = np.stack([embed_osnet([c], osnet, device)[0] for c in demo_crops])
    print(f"  OSNet shape: WA={wa_osnet.shape}, Demo={demo_osnet.shape}")

    # B: Multi-scale DINOv2 (full + upper + lower + jersey)
    print("  [B] Multi-scale DINOv2 embedding...")
    wa_multiscale = np.stack([
        embed_multiscale(c, dinov2, device) for c in wa_crops
    ])
    demo_multiscale = np.stack([
        embed_multiscale(c, dinov2, device) for c in demo_crops
    ])
    print(f"  Multi-scale shape: WA={wa_multiscale.shape}, Demo={demo_multiscale.shape}")

    # C: Standard DINOv2 (baseline for whitening comparison)
    print("  [C] Standard DINOv2 embedding (for whitening)...")
    wa_dinov2 = np.stack([embed_crops([c], dinov2, device)[0] for c in wa_crops])
    demo_dinov2 = np.stack([embed_crops([c], dinov2, device)[0] for c in demo_crops])
    print(f"  DINOv2 shape: WA={wa_dinov2.shape}, Demo={demo_dinov2.shape}")

    # C2: PCA-whitened DINOv2
    print("  [C2] PCA-whitening DINOv2 embeddings...")
    # Fit PCA on WA dataset, apply to both
    wa_whitened = pca_whiten(wa_dinov2, n_components=64)
    demo_whitened = pca_whiten(demo_dinov2, n_components=64)
    print(f"  Whitened shape: WA={wa_whitened.shape}, Demo={demo_whitened.shape}")

    # D: Fusion — concatenate OSNet + multi-scale DINOv2 (normalized)
    print("  [D] Fusion: OSNet + multi-scale DINOv2...")
    wa_fusion = np.concatenate([wa_osnet, wa_multiscale], axis=1)
    norms = np.linalg.norm(wa_fusion, axis=1, keepdims=True) + 1e-8
    wa_fusion = wa_fusion / norms

    demo_fusion = np.concatenate([demo_osnet, demo_multiscale], axis=1)
    norms = np.linalg.norm(demo_fusion, axis=1, keepdims=True) + 1e-8
    demo_fusion = demo_fusion / norms
    print(f"  Fusion shape: WA={wa_fusion.shape}, Demo={demo_fusion.shape}")

    # E: Whitened fusion (whitened DINOv2 + OSNet)
    print("  [E] Fusion: OSNet + whitened DINOv2...")
    wa_fusion2 = np.concatenate([wa_osnet, wa_whitened], axis=1)
    norms = np.linalg.norm(wa_fusion2, axis=1, keepdims=True) + 1e-8
    wa_fusion2 = wa_fusion2 / norms

    demo_fusion2 = np.concatenate([demo_osnet, demo_whitened], axis=1)
    norms = np.linalg.norm(demo_fusion2, axis=1, keepdims=True) + 1e-8
    demo_fusion2 = demo_fusion2 / norms
    print(f"  Fusion2 shape: WA={wa_fusion2.shape}, Demo={demo_fusion2.shape}")

    # ── Evaluate on WA Open (20 players) ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== WA OPEN (20 players, 696 photos) ===")
    print("=" * 70)

    approaches = [
        ("A: OSNet Re-ID", wa_osnet),
        ("B: Multi-scale DINOv2", wa_multiscale),
        ("C: DINOv2 baseline", wa_dinov2),
        ("C2: DINOv2 whitened", wa_whitened),
        ("D: OSNet+MultiScale fusion", wa_fusion),
        ("E: OSNet+Whitened fusion", wa_fusion2),
    ]

    wa_results = {}
    for name, embs in approaches:
        print(f"\n  --- {name} ---")
        _, metrics = sweep_thresholds(embs, wa_fnames, true_player_wa, 20, name)
        wa_results[name] = metrics
        stats = compute_embedding_stats(embs, wa_fnames, true_player_wa, name)
        wa_results[name]['stats'] = stats

    # ── Evaluate on Demo (5 players) ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== DEMO (5 players, 171 photos) ===")
    print("=" * 70)

    demo_approaches = [
        ("A: OSNet Re-ID", demo_osnet),
        ("B: Multi-scale DINOv2", demo_multiscale),
        ("C: DINOv2 baseline", demo_dinov2),
        ("C2: DINOv2 whitened", demo_whitened),
        ("D: OSNet+MultiScale fusion", demo_fusion),
        ("E: OSNet+Whitened fusion", demo_fusion2),
    ]

    demo_results = {}
    for name, embs in demo_approaches:
        print(f"\n  --- {name} ---")
        _, metrics = sweep_thresholds(embs, demo_fnames, true_player_demo, 5, name)
        demo_results[name] = metrics

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== SUMMARY TABLE ===")
    print("=" * 70)
    print(f"\n{'Approach':<30} {'WA ARI':>8} {'WA Cov':>8} {'WA Pur':>8} {'WA Comp':>9}  {'Demo ARI':>9} {'Demo Cov':>9}")
    print("-" * 80)
    for name, _ in approaches:
        wr = wa_results[name]
        dr = demo_results[name]
        print(
            f"  {name:<28} {wr['ari']:>8.3f} {wr['covered']:>4}/{wr['n_true']} "
            f"{wr['purity']:>8.3f} {wr['composite']:>9.3f}  "
            f"{dr['ari']:>9.3f}  {dr['covered']:>4}/{dr['n_true']}"
        )

    # ── Best approach analysis ─────────────────────────────────────────────────
    best_name = max(wa_results, key=lambda k: wa_results[k]['composite'])
    best = wa_results[best_name]
    print(f"\n*** Best on WA Open (composite metric): {best_name} ***")
    print(f"    Composite: {best['composite']:.3f}  ARI: {best['ari']:.3f}")
    print(f"    Coverage: {best['covered']}/{best['n_true']}  Purity: {best['purity']:.3f}")

    # ── Theoretical limit analysis ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== THEORETICAL LIMIT ANALYSIS ===")
    print("=" * 70)
    best_embs_name, best_embs = max(approaches, key=lambda kv: wa_results[kv[0]]['composite'])
    stats = wa_results[best_embs_name]['stats']
    ratio = stats['ratio']
    print(f"\nBest approach: {best_embs_name}")
    print(f"  Intra/inter ratio: {ratio:.4f}")
    if ratio < 0.5:
        print("  -> GOOD: Players are well-separated (ratio < 0.5)")
        print("     Theoretically achievable to distinguish all 20 players from appearance alone.")
    elif ratio < 1.0:
        print("  -> MARGINAL: Some separation exists (ratio < 1.0)")
        print("     Players are partially distinguishable but overlap in embedding space.")
    else:
        print("  -> POOR: Players cannot be reliably separated (ratio >= 1.0)")
        print("     Intra-player variance exceeds inter-player variance — appearance alone insufficient.")

    print("\n  5 most confused player pairs (hardest to separate):")
    for dist, p1, p2 in stats['pair_dists'][:5]:
        print(f"    Players {p1:2d} & {p2:2d}: cosine dist = {dist:.4f}")

    print("\n  Interpretation:")
    n_distinguishable = sum(1 for d, _, _ in stats['pair_dists'] if d > 0.10)
    n_total_pairs = len(stats['pair_dists'])
    print(f"    {n_distinguishable}/{n_total_pairs} player pairs have cosine dist > 0.10 "
          f"(reasonably distinguishable)")

    print("\nDone.")


if __name__ == '__main__':
    main()

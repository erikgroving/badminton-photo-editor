"""
Unsupervised clustering experiment for culling.

Pipeline:
  1. Extract CNN embeddings for all labeled raws (reuses best culling backbone).
  2. PCA → 64 dims, then K-means into K clusters.
  3. For each cluster, train a logistic-regression binary classifier
     (fast: no GPU needed, just sklearn on the embeddings).
  4. A meta-learner (small MLP) combines the K cluster-specialist scores
     with the raw cluster assignment into a final keep/reject decision.
  5. Compare vs the plain backbone to see if specialisation helps.

The clusters tend to align with scene-level categories (e.g. wide-court vs
close-up, brightly-lit vs indoor-dark, singles vs doubles coverage) rather
than exactly "blur" vs "expression" — but the per-cluster classifiers can
still specialise on whatever makes a shot "good" within that visual context.

Usage:
    python -m models.culling.cluster_culling
    python -m models.culling.cluster_culling --k 8 --backbone efficientnet_b3
    python -m models.culling.cluster_culling --visualise   # saves UMAP plot

Checkpoint: checkpoints/culling_cluster_k<K>_<backbone>.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    CHECKPOINTS_DIR, CULL_BATCH_SIZE, CULL_FBETA, CULL_FN_WEIGHT,
    MAPPING_FILE, THUMB_SIZE,
)
from data.mapping import flat_entries, load_mapping
from data.raw_reader import extract_thumbnail
from models.culling.train import _evaluate_logits, _print_metrics, fbeta

_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class _SimpleDS(torch.utils.data.Dataset):
    def __init__(self, entries):
        self.entries = entries
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx):
        e = self.entries[idx]
        img = extract_thumbnail(e["raw"], size=THUMB_SIZE)
        return _TF(img), float(e["label"])


def _extract_embeddings(backbone: str, entries: list[dict],
                        batch_size: int, device: torch.device) -> np.ndarray:
    """Returns (N, feat_dim) float32 embeddings."""
    import timm
    model = timm.create_model(backbone, pretrained=True,
                               num_classes=0, global_pool="avg").to(device)
    # Load fine-tuned weights if checkpoint exists
    ckpt_candidates = sorted(
        CHECKPOINTS_DIR.glob(f"culling_{backbone.replace('/', '_')}*.pt"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if ckpt_candidates:
        ck = torch.load(str(ckpt_candidates[0]), map_location=device,
                        weights_only=False)
        # Extract backbone weights only
        state = {k.replace("cnn.", ""): v
                 for k, v in ck["model_state"].items() if k.startswith("cnn.")}
        if state:
            model.load_state_dict(state, strict=False)
            print(f"  Loaded fine-tuned weights from {ckpt_candidates[0].name}")
        else:
            state2 = {k: v for k, v in ck["model_state"].items()
                      if not k.startswith(("fusion", "attr_mlp"))}
            model.load_state_dict(state2, strict=False)
            print(f"  Loaded backbone weights from {ckpt_candidates[0].name}")
    else:
        print(f"  No fine-tuned checkpoint found — using pretrained ImageNet weights")

    model.eval()
    loader = DataLoader(_SimpleDS(entries), batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
    all_embs = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="  Extracting embeddings", leave=False):
            all_embs.append(model(imgs.to(device)).cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def _cluster_and_train(
    train_embs, train_labels,
    val_embs,   val_labels,
    test_embs,  test_labels,
    k: int, fn_weight: float,
) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    print(f"\nPCA 64 dims...")
    scaler = StandardScaler()
    pca    = PCA(n_components=min(64, train_embs.shape[1]), random_state=42)
    train_pca = pca.fit_transform(scaler.fit_transform(train_embs))
    val_pca   = pca.transform(scaler.transform(val_embs))
    test_pca  = pca.transform(scaler.transform(test_embs))
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    print(f"K-means k={k}...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    train_clusters = km.fit_predict(train_pca)
    val_clusters   = km.predict(val_pca)
    test_clusters  = km.predict(test_pca)

    # Per-cluster class balance
    print(f"\nCluster sizes and keep rates:")
    for c in range(k):
        mask  = train_clusters == c
        n     = mask.sum()
        krate = train_labels[mask].mean() if n > 0 else 0
        print(f"  cluster {c}: {n:5d} samples,  keep rate {krate:.1%}")

    # Per-cluster logistic regression specialists
    specialists = []
    for c in range(k):
        mask = train_clusters == c
        if mask.sum() < 10:
            specialists.append(None)
            continue
        lr = LogisticRegression(
            class_weight={0: 1.0, 1: fn_weight},
            max_iter=500, C=1.0, random_state=42,
        )
        lr.fit(train_pca[mask], train_labels[mask].astype(int))
        specialists.append(lr)

    def _specialist_scores(embs_pca, cluster_ids):
        """Returns (N, k) matrix of each specialist's keep probability."""
        scores = np.zeros((len(embs_pca), k), dtype=np.float32)
        for c, spec in enumerate(specialists):
            if spec is None:
                scores[:, c] = 0.5
                continue
            proba = spec.predict_proba(embs_pca)
            pos_col = list(spec.classes_).index(1) if 1 in spec.classes_ else -1
            scores[:, c] = proba[:, pos_col] if pos_col >= 0 else 0.0
        return scores

    # Meta-learner: logistic regression on (specialist_scores, cluster_onehot)
    train_spec = _specialist_scores(train_pca, train_clusters)
    train_oh   = np.eye(k)[train_clusters]
    train_meta = np.concatenate([train_spec, train_oh], axis=1)

    val_spec   = _specialist_scores(val_pca, val_clusters)
    val_oh     = np.eye(k)[val_clusters]
    val_meta   = np.concatenate([val_spec, val_oh], axis=1)

    test_spec  = _specialist_scores(test_pca, test_clusters)
    test_oh    = np.eye(k)[test_clusters]
    test_meta  = np.concatenate([test_spec, test_oh], axis=1)

    print(f"\nTraining meta-learner on cluster specialist scores...")
    meta = LogisticRegression(
        class_weight={0: 1.0, 1: fn_weight},
        max_iter=1000, C=0.5, random_state=42,
    )
    meta.fit(train_meta, train_labels.astype(int))

    def _eval_set(embs_meta, labels_arr, split_name):
        proba   = meta.predict_proba(embs_meta)
        pos_col = list(meta.classes_).index(1)
        logits  = torch.tensor(
            np.log(proba[:, pos_col] / (1 - proba[:, pos_col] + 1e-9)),
            dtype=torch.float32,
        )
        labs = torch.tensor(labels_arr, dtype=torch.float32)
        m = _evaluate_logits(logits, labs, fn_weight, CULL_FBETA)
        print(f"  [{split_name}]", end="")
        _print_metrics(m, CULL_FBETA)
        return m

    val_m  = _eval_set(val_meta,  val_labels,  "val")
    test_m = _eval_set(test_meta, test_labels, "test")

    return {
        "k": k,
        "val_metrics":  val_m,
        "test_metrics": test_m,
        "kmeans":    km,
        "pca":       pca,
        "scaler":    scaler,
        "specialists": specialists,
        "meta":      meta,
    }


def _save_umap(train_embs, train_labels, train_clusters, k, out_path):
    """Optional UMAP visualisation — only runs if umap-learn is installed."""
    try:
        import umap
        import matplotlib.pyplot as plt

        print("Running UMAP (this may take a few minutes)...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        emb2d   = reducer.fit_transform(train_embs[:5000])   # cap for speed
        labels  = train_labels[:5000]
        clusters = train_clusters[:5000]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sc0 = axes[0].scatter(emb2d[:, 0], emb2d[:, 1], c=labels,
                               cmap="RdYlGn", s=3, alpha=0.5)
        axes[0].set_title("Kept (green) vs Culled (red)")
        plt.colorbar(sc0, ax=axes[0])

        sc1 = axes[1].scatter(emb2d[:, 0], emb2d[:, 1], c=clusters,
                               cmap="tab10", s=3, alpha=0.5)
        axes[1].set_title(f"K={k} clusters")
        plt.colorbar(sc1, ax=axes[1])

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"UMAP saved: {out_path}")
    except ImportError:
        print("umap-learn not installed — skipping visualisation")
        print("  Install with:  pip install umap-learn matplotlib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",   type=str, default="efficientnet_b0")
    parser.add_argument("--k",          type=int, default=6,
                        help="Number of clusters")
    parser.add_argument("--fn-weight",  type=float, default=CULL_FN_WEIGHT)
    parser.add_argument("--batch-size", type=int, default=CULL_BATCH_SIZE)
    parser.add_argument("--visualise",  action="store_true",
                        help="Save UMAP scatter plot (requires umap-learn)")
    args = parser.parse_args()

    mapping       = load_mapping()
    train_entries = [e for e in flat_entries(mapping, split="train")
                     if e["label"] is not None and e["raw"]]
    val_entries   = [e for e in flat_entries(mapping, split="val")
                     if e["label"] is not None and e["raw"]]
    test_entries  = [e for e in flat_entries(mapping, split="test")
                     if e["label"] is not None and e["raw"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting embeddings with {args.backbone} ({device})...")

    train_embs = _extract_embeddings(args.backbone, train_entries, args.batch_size, device)
    val_embs   = _extract_embeddings(args.backbone, val_entries,   args.batch_size, device)
    test_embs  = _extract_embeddings(args.backbone, test_entries,  args.batch_size, device)

    train_labels = np.array([e["label"] for e in train_entries], dtype=np.float32)
    val_labels   = np.array([e["label"] for e in val_entries],   dtype=np.float32)
    test_labels  = np.array([e["label"] for e in test_entries],  dtype=np.float32)

    results = _cluster_and_train(
        train_embs, train_labels,
        val_embs,   val_labels,
        test_embs,  test_labels,
        k=args.k, fn_weight=args.fn_weight,
    )

    if args.visualise:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        sc  = results["scaler"]
        pca = results["pca"]
        km  = results["kmeans"]
        tr_pca  = pca.transform(sc.transform(train_embs))
        tr_clus = km.predict(tr_pca)
        out = CHECKPOINTS_DIR.parent / "sanity_check" / f"umap_k{args.k}.png"
        out.parent.mkdir(exist_ok=True)
        _save_umap(train_embs, train_labels, tr_clus, args.k, out)

    # Save everything for inference
    import pickle
    safe = args.backbone.replace("/", "_")
    ckpt = CHECKPOINTS_DIR / f"culling_cluster_k{args.k}_{safe}.pkl"
    with open(ckpt, "wb") as f:
        pickle.dump(results, f)
    print(f"\nCluster model saved: {ckpt}")

    # Summary comparison
    vm = results["val_metrics"]
    print(f"\nCluster ensemble (k={args.k}) val results:")
    print(f"  recall={vm['recall']:.1%}  selection={vm['selection_rate']:.1%}"
          f"  F2={vm.get('f2', 0):.4f}  asym_cost={vm['asym_cost']:.0f}")


if __name__ == "__main__":
    main()

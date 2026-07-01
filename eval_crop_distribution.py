"""
Distribution evaluation: are model crops indistinguishable from Jay's crops?

Method
------
1. Run the trained crop model to get predicted box+angle for every image.
2. Extract the predicted crop patch and the GT crop patch from each raw thumbnail.
3. Embed both sets with a frozen backbone (DINOv2-base, no fine-tuning).
4. Train a logistic-regression probe on train embeddings (Jay=1 vs Model=0).
5. Report test-set accuracy.
   - ~50%  → distributions overlap, classifier can't tell them apart → great
   - ~90%+ → clear gap between Jay's crops and model crops → needs work
6. Also report MMD (Maximum Mean Discrepancy) — lower = more similar.
7. Save a UMAP scatter plot to logs/crop_distribution_umap.png.

Usage:
    python eval_crop_distribution.py
    python eval_crop_distribution.py --split test          # evaluate test only
    python eval_crop_distribution.py --no-umap             # skip UMAP (faster)
    python eval_crop_distribution.py --ckpt checkpoints/cropping_angle_dinov2_gan.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINTS_DIR, CROP_GT_FILE
from data.raw_reader import extract_thumbnail_ar
from models.cropping.model import ANGLE_SCALE, build_crop_model

CROP_CKPT      = CHECKPOINTS_DIR / "cropping_angle_vit_base_patch14_reg4_dinov2.pt"
EMBED_MODEL    = "vit_small_patch14_reg4_dinov2"   # lightweight DINOv2 for embeddings
EXTRACT_SIZE   = 512
PATCH_SIZE     = 224
LOG_DIR        = Path("logs")


# ── Crop extraction utilities ─────────────────────────────────────────────────

def extract_gt_crop(img_pil: Image.Image, box: list[float], angle_deg: float,
                    out_size: int) -> Image.Image:
    """Extract and rotate the GT crop patch from a PIL raw thumbnail."""
    w, h = img_pil.size
    x1, y1, x2, y2 = box
    region = img_pil.crop((int(x1*w), int(y1*h), int(x2*w), int(y2*h)))
    if abs(angle_deg) > 45.0:
        region = region.rotate(round(angle_deg), expand=True, resample=Image.BILINEAR)
    return region.resize((out_size, out_size), Image.BILINEAR)


def extract_model_crop(img_pil: Image.Image, box_pred: list[float], angle_pred_deg: float,
                       out_size: int) -> Image.Image:
    """Extract and rotate the model-predicted crop patch."""
    return extract_gt_crop(img_pil, box_pred, angle_pred_deg, out_size)


# ── Paired dataset ────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """
    Returns (gt_crop, model_crop) pairs so we can batch-embed efficiently.
    gt_crop  label = 1  (Jay)
    model_crop label = 0  (model)
    """
    def __init__(self, records: list[dict], predictions: list[dict], tf):
        assert len(records) == len(predictions)
        self.records     = records
        self.predictions = predictions
        self.tf          = tf

    def __len__(self) -> int:
        return len(self.records) * 2   # one real + one model per pair

    def __getitem__(self, idx: int):
        i       = idx % len(self.records)
        is_real = idx < len(self.records)
        r       = self.records[i]
        p       = self.predictions[i]
        raw_pil = extract_thumbnail_ar(r["raw"], max_size=EXTRACT_SIZE)

        if is_real:
            patch = extract_gt_crop(raw_pil, r["box"], r.get("angle_deg", 0.0), PATCH_SIZE)
            label = 1
        else:
            patch = extract_model_crop(raw_pil, p["box"], p["angle_deg"], PATCH_SIZE)
            label = 0

        return self.tf(patch), label


# ── Run crop model on all records ─────────────────────────────────────────────

class InferDataset(Dataset):
    def __init__(self, records: list[dict], tf):
        self.records = records
        self.tf      = tf

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r   = self.records[idx]
        img = extract_thumbnail_ar(r["raw"], max_size=EXTRACT_SIZE)
        return self.tf(img)


def run_crop_model(records: list[dict], ckpt_path: Path,
                   device: torch.device) -> list[dict]:
    """Return predicted {box, angle_deg} for every record."""
    import timm as _timm
    ck    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_crop_model(backbone="vit_base_patch14_reg4_dinov2",
                             pretrained=False, use_angle_head=True)
    model.load_state_dict(ck["model_state"])
    model = model.to(device).eval()
    model.set_grad_checkpointing(enable=False)

    data_cfg  = _timm.data.resolve_model_data_config(model.backbone)
    inp_size  = data_cfg.get("input_size", (3, 224, 224))[1]
    norm_mean = list(data_cfg.get("mean", (0.485, 0.456, 0.406)))
    norm_std  = list(data_cfg.get("std",  (0.229, 0.224, 0.225)))
    tf = transforms.Compose([
        transforms.Resize((inp_size, inp_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    loader = DataLoader(InferDataset(records, tf),
                        batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Running crop model"):
            box_pred, angle_pred = model(batch.to(device))
            for b, a in zip(box_pred.cpu().tolist(), angle_pred.cpu().tolist()):
                preds.append({"box": b, "angle_deg": float(a) * ANGLE_SCALE})

    return preds


# ── Embed crops ───────────────────────────────────────────────────────────────

def embed_crops(records: list[dict], predictions: list[dict],
                device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (embeddings, labels) where embeddings is (2N, D) and labels is (2N,).
    First N rows = GT crops (label=1), next N rows = model crops (label=0).
    """
    import timm as _timm
    embed_model = _timm.create_model(EMBED_MODEL, pretrained=True,
                                     num_classes=0, global_pool="avg").to(device).eval()
    data_cfg    = _timm.data.resolve_data_config({}, model=EMBED_MODEL)
    inp_size    = data_cfg.get("input_size", (3, 224, 224))[1]
    norm_mean   = list(data_cfg.get("mean", (0.485, 0.456, 0.406)))
    norm_std    = list(data_cfg.get("std",  (0.229, 0.224, 0.225)))
    tf = transforms.Compose([
        transforms.Resize((inp_size, inp_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    ds     = PairDataset(records, predictions, tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    all_emb, all_lab = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting embeddings"):
            emb = embed_model(imgs.to(device))
            all_emb.append(emb.cpu().numpy())
            all_lab.extend(labels.tolist())

    return np.concatenate(all_emb), np.array(all_lab)


# ── Metrics ───────────────────────────────────────────────────────────────────

def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """Unbiased Maximum Mean Discrepancy with RBF kernel."""
    from sklearn.metrics.pairwise import rbf_kernel
    n, m = len(X), len(Y)
    kxx   = rbf_kernel(X, X, gamma=gamma)
    kyy   = rbf_kernel(Y, Y, gamma=gamma)
    kxy   = rbf_kernel(X, Y, gamma=gamma)
    return float((kxx.sum() - np.trace(kxx)) / (n*(n-1))
               + (kyy.sum() - np.trace(kyy)) / (m*(m-1))
               - 2 * kxy.mean())


def classifier_accuracy(emb_train: np.ndarray, lab_train: np.ndarray,
                        emb_test: np.ndarray,  lab_test: np.ndarray) -> tuple[float, float]:
    """Train a logistic regression probe; return (train_acc, test_acc)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(emb_train)
    X_te   = scaler.transform(emb_test)
    clf    = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_tr, lab_train)
    return clf.score(X_tr, lab_train), clf.score(X_te, lab_test)


# ── UMAP visualization ────────────────────────────────────────────────────────

def make_umap(embeddings: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    try:
        from umap import UMAP
    except ImportError:
        try:
            from umap.umap_ import UMAP
        except ImportError:
            print("  UMAP not installed — skipping visualization. (pip install umap-learn)")
            return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("  Running UMAP (may take 1-2 min)…")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    xy      = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 7))
    for lbl, color, name in [(1, "#2196F3", "Jay (GT)"), (0, "#FF5722", "Model")]:
        mask = labels == lbl
        ax.scatter(xy[mask, 0], xy[mask, 1], c=color, alpha=0.4,
                   s=10, label=name, rasterized=True)

    ax.set_title("Crop distribution: Jay crops vs model crops\n"
                 "(overlap = indistinguishable, clusters = gap)", fontsize=13)
    ax.legend(markerscale=3, fontsize=11)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.axis("off")
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved UMAP: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    if not CROP_GT_FILE.exists():
        raise FileNotFoundError(f"Need {CROP_GT_FILE} — run data/crop_detector.py first.")

    with open(CROP_GT_FILE) as fh:
        all_records = json.load(fh)

    split = args.split
    if split == "all":
        records = all_records
    else:
        records = [r for r in all_records if r["split"] == split]
    print(f"Using {len(records):,} records ({split} split)")

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt     = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"Crop checkpoint not found: {ckpt}")
    print(f"Crop model: {ckpt.name}")

    # 1. Run crop model
    print("\nStep 1: Running crop model…")
    predictions = run_crop_model(records, ckpt, device)

    # 2. Embed both distributions
    print("\nStep 2: Embedding crops…")
    embeddings, labels = embed_crops(records, predictions, device)

    real_emb  = embeddings[labels == 1]
    model_emb = embeddings[labels == 0]
    print(f"  Real (Jay) crops embedded:  {len(real_emb):,}")
    print(f"  Model crops embedded:       {len(model_emb):,}")

    # Use split-aware train/test division from the original records
    # For the probe, train on records[split==train] and test on records[split==test]
    n = len(records)
    is_train = np.array([r["split"] == "train" for r in records])

    # labels are interleaved: [real_0..real_N-1, model_0..model_N-1]
    real_idx  = np.arange(n)
    model_idx = np.arange(n, 2*n)

    train_idx = np.concatenate([real_idx[is_train],  model_idx[is_train]])
    test_idx  = np.concatenate([real_idx[~is_train], model_idx[~is_train]])

    emb_train = embeddings[train_idx]
    lab_train = labels[train_idx]
    emb_test  = embeddings[test_idx]
    lab_test  = labels[test_idx]

    # 3. Classifier probe
    print("\nStep 3: Training classifier probe…")
    train_acc, test_acc = classifier_accuracy(emb_train, lab_train, emb_test, lab_test)
    gap = test_acc - 0.5

    # 4. MMD (subsample if large)
    print("Step 4: Computing MMD…")
    max_mmd  = 2000
    re_sub   = real_emb [:max_mmd] if len(real_emb)  > max_mmd else real_emb
    mo_sub   = model_emb[:max_mmd] if len(model_emb) > max_mmd else model_emb
    mmd_val  = mmd_rbf(re_sub, mo_sub)

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CROP DISTRIBUTION EVALUATION  ({ckpt.name})")
    print(f"{'='*60}")
    print(f"  Classifier probe — train acc: {train_acc:.1%}")
    print(f"  Classifier probe — test  acc: {test_acc:.1%}  "
          f"({'~random = great' if gap < 0.05 else 'gap = ' + f'{gap:+.1%}'})")
    print(f"  MMD (lower = more similar): {mmd_val:.6f}")
    if test_acc < 0.55:
        verdict = "EXCELLENT — distributions overlap, Jay can't be told from model"
    elif test_acc < 0.65:
        verdict = "GOOD — mostly overlapping, small detectable gap"
    elif test_acc < 0.75:
        verdict = "FAIR — noticeable gap, model crops differ from Jay's"
    else:
        verdict = "POOR — distributions are clearly separated"
    print(f"\n  Verdict: {verdict}")
    print(f"{'='*60}")

    # 5. UMAP
    if not args.no_umap:
        print("\nStep 5: UMAP visualization…")
        # If all records used, subsample for UMAP speed
        max_umap = 3000
        if len(embeddings) > max_umap:
            rng   = np.random.default_rng(42)
            idx_u = rng.choice(len(embeddings), max_umap, replace=False)
            emb_u, lab_u = embeddings[idx_u], labels[idx_u]
        else:
            emb_u, lab_u = embeddings, labels
        out_png = LOG_DIR / f"crop_distribution_umap_{ckpt.stem}.png"
        make_umap(emb_u, lab_u, out_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    default=str(CROP_CKPT))
    parser.add_argument("--split",   default="all",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--no-umap", action="store_true")
    args = parser.parse_args()
    main(args)

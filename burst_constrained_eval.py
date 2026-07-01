"""
Evaluate culling models with a burst constraint: at most 1 selection per 1-second burst.
Within each burst, only the highest-scoring photo can be selected.
"""
import sys
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINTS_DIR, CULL_BATCH_SIZE, THUMB_SIZE
from data.mapping import flat_entries, load_mapping, _read_cr3_ts
from data.raw_reader import extract_thumbnail
from models.culling.model import build_model
from models.culling.train import CullingDataset

TARGETS = [0.70, 0.80, 0.90, 0.95, 0.98]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

MODELS = [
    ("efficientnet_b0",           "vit_base_patch14_reg4_dinov2", None),
    ("efficientnet_b3",           "vit_base_patch14_reg4_dinov2", None),
    ("siglip_512 [base]",         "vit_base_patch16_siglip_512",  None),
    ("dinov2 [base+reg]",         "vit_base_patch14_reg4_dinov2", None),
    ("dinov2 [large]",            "vit_large_patch14_dinov2",     None),
]

# Simpler: just do all checkpoints that exist
CHECKPOINTS = [
    ("efficientnet_b0",        "efficientnet_b0"),
    ("efficientnet_b3",        "efficientnet_b3"),
    ("siglip_512",             "vit_base_patch16_siglip_512"),
    ("dinov2_base",            "vit_base_patch14_reg4_dinov2"),
    ("dinov2_large",           "vit_large_patch14_dinov2"),
]


def get_burst_key(raw_path: str) -> str:
    ts = _read_cr3_ts(raw_path)
    folder = Path(raw_path).parent.name
    if ts:
        return f"{folder}|{ts.split('.')[0]}"
    return f"{folder}|{Path(raw_path).stem}"  # fallback: treat each file as its own burst


def load_val_entries():
    mapping = load_mapping()
    return [e for e in flat_entries(mapping, split="val")
            if e["label"] is not None and e["raw"]]


def get_logits(backbone, ckpt_path, val_entries, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    input_size = ck.get("input_size", THUMB_SIZE[0])
    norm_mean  = ck.get("norm_mean",  _IMAGENET_MEAN)
    norm_std   = ck.get("norm_std",   _IMAGENET_STD)

    resize = [transforms.Resize((input_size, input_size))] if input_size != THUMB_SIZE[0] else []
    tf = transforms.Compose(resize + [
        transforms.ToTensor(),
        transforms.Normalize(list(norm_mean), list(norm_std)),
    ])

    ds = CullingDataset(val_entries, tf)
    loader = DataLoader(ds, batch_size=CULL_BATCH_SIZE * 2,
                        shuffle=False, num_workers=4, pin_memory=True)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            all_logits.append(model(imgs.to(device)).squeeze(1).cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def apply_burst_constraint(probs, labels, burst_keys):
    """
    Within each burst, only allow the highest-scoring photo to be selected.
    Returns modified probs where non-top photos in each burst are zeroed out.
    """
    probs = probs.clone()
    burst_indices = defaultdict(list)
    for i, key in enumerate(burst_keys):
        burst_indices[key].append(i)

    for key, idxs in burst_indices.items():
        if len(idxs) == 1:
            continue
        burst_probs = probs[idxs]
        best = int(burst_probs.argmax())
        for j, idx in enumerate(idxs):
            if j != best:
                probs[idx] = 0.0  # suppress non-top photos
    return probs


def eval_at_targets(probs, labels, targets):
    # Pre-compute all thresholds once
    thresholds = [float(t) for t in torch.linspace(0.001, 0.999, 500)]
    cache = {}
    for t in thresholds:
        preds = (probs >= t).float()
        tp = int(((preds == 1) & (labels == 1)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sel = float(preds.sum()) / len(preds)
        cache[t] = (recall, sel)

    results = []
    for target in targets:
        # Highest threshold (smallest selection) that still achieves target recall
        best_t, best_sel, best_actual = None, None, None
        for t in thresholds:
            recall, sel = cache[t]
            if recall >= target:
                if best_t is None or t > best_t:
                    best_t, best_sel, best_actual = t, sel, recall
        if best_t is None:
            # No threshold achieves this recall — report max achievable
            max_recall = max(r for r, s in cache.values())
            results.append((None, max_recall))
        else:
            results.append((best_sel, best_actual))
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_entries = load_val_entries()
    print(f"Val entries: {len(val_entries):,}  |  device: {device}")

    print("Reading burst keys from CR3 timestamps...")
    burst_keys = [get_burst_key(e["raw"]) for e in val_entries]
    n_bursts = len(set(burst_keys))
    print(f"Unique bursts: {n_bursts:,}  (avg {len(val_entries)/n_bursts:.1f} photos/burst)")
    print()

    target_labels = [f"R{int(t*100)}%" for t in TARGETS]
    header = f"  {'Model':<22}  {'Mode':<14}" + "".join(f"  {lbl:>12}" for lbl in target_labels)
    print(header)
    print("  " + "-" * (22 + 16 + 14 * len(TARGETS)))

    for label, backbone in CHECKPOINTS:
        ckpt_path = CHECKPOINTS_DIR / f"culling_{backbone.replace('/', '_')}.pt"
        if not ckpt_path.exists():
            continue

        logits, labels = get_logits(backbone, str(ckpt_path), val_entries, device)
        probs = torch.sigmoid(logits)

        # Unconstrained
        def fmt(sel, act):
            if sel is None:
                return f"N/A (max {act:.0%})"
            return f"{sel:.1%} ({act:.0%})"

        raw_results = eval_at_targets(probs, labels, TARGETS)
        row = f"  {label:<22}  {'unconstrained':<14}" + "".join(
            f"  {fmt(sel, act):>16}" for sel, act in raw_results)
        print(row)

        # Burst-constrained
        probs_c = apply_burst_constraint(probs, labels, burst_keys)
        con_results = eval_at_targets(probs_c, labels, TARGETS)
        row = f"  {'':<22}  {'burst-max-1':<14}" + "".join(
            f"  {fmt(sel, act):>16}" for sel, act in con_results)
        print(row)
        print()

    print("Format: selection_rate (actual_recall)")
    print("burst-max-1: within each 1-second burst, only the top-scoring photo can be selected")


if __name__ == "__main__":
    main()

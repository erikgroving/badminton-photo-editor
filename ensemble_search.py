"""
Exhaustive ensemble configuration search.

Tries every non-empty subset of the four trained backbones, plus a grid of
weights for the top-performing pairs/triples, and reports the minimum selection
rate needed to hit each recall target.

Usage:
    python ensemble_search.py
"""
import itertools
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINTS_DIR, CULL_BATCH_SIZE, THUMB_SIZE
from data.mapping import flat_entries, load_mapping
from models.culling.model import build_model
from models.culling.train import CullingDataset

TARGETS = [0.70, 0.80, 0.90, 0.95, 0.98]

BACKBONES = {
    "b0":  "efficientnet_b0",
    "b3":  "efficientnet_b3",
    "r50": "resnet50",
    "mv3": "mobilenetv3_large_100",
}

_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_all_logits(val_entries: list, device: torch.device) -> dict[str, torch.Tensor]:
    """Load logits for every backbone. Returns {short_name: logit_tensor}."""
    result = {}
    for short, backbone in BACKBONES.items():
        ckpt_path = CHECKPOINTS_DIR / f"culling_{backbone.replace('/', '_')}.pt"
        if not ckpt_path.exists():
            print(f"  [skip] {backbone} — no checkpoint")
            continue
        ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = build_model(backbone=backbone, pretrained=False).to(device)
        model.load_state_dict(ck["model_state"])
        model.eval()
        ds = CullingDataset(val_entries, _TF)
        loader = DataLoader(ds, batch_size=CULL_BATCH_SIZE * 2,
                            shuffle=False, num_workers=4, pin_memory=True)
        logits_list = []
        with torch.no_grad():
            for imgs, _ in tqdm(loader, desc=f"  {short}", leave=False):
                logits_list.append(model(imgs.to(device)).squeeze(1).cpu())
        result[short] = torch.cat(logits_list)
        print(f"  Loaded {short} ({backbone})")
    return result


def min_selection_for_recall(probs: torch.Tensor, labels: torch.Tensor,
                              target: float, n_steps: int = 1000) -> tuple[float, float]:
    """Return (min_selection, actual_recall) for the highest threshold hitting target."""
    best_t = 0.0
    for t in torch.linspace(0.001, 0.999, n_steps):
        t_f = float(t)
        preds = (probs >= t_f).float()
        tp = int(((preds == 1) & (labels == 1)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        if tp + fn > 0 and tp / (tp + fn) >= target:
            best_t = t_f
    preds = (probs >= best_t).float()
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sel    = (tp + fp) / len(labels)
    return sel, recall


def eval_probs(probs: torch.Tensor, labels: torch.Tensor) -> list[tuple[float, float]]:
    return [min_selection_for_recall(probs, labels, t) for t in TARGETS]


def weighted_probs(logits_dict: dict, weights: dict) -> torch.Tensor:
    """Weighted average of sigmoid(logits). weights keys must match logits_dict keys."""
    total_w = sum(weights.values())
    p = sum(torch.sigmoid(logits_dict[k]) * w for k, w in weights.items())
    return p / total_w


def fmt_cells(results: list[tuple[float, float]]) -> str:
    return "  ".join(f"{s:.1%}({r:.0%})" for s, r in results)


def main():
    mapping     = load_mapping()
    val_entries = [e for e in flat_entries(mapping, split="val")
                   if e["label"] is not None and e["raw"]]
    labels      = torch.tensor([e["label"] for e in val_entries], dtype=torch.float32)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nVal set: {len(val_entries):,}  |  device: {device}")
    print(f"Loading all backbone logits...\n")
    logits = load_all_logits(val_entries, device)

    keys = list(logits.keys())

    # ── Individual models ────────────────────────────────────────────────────────
    rows: list[tuple[str, list]] = []

    for k in keys:
        probs = torch.sigmoid(logits[k])
        rows.append((k, eval_probs(probs, labels)))

    # ── Equal-weight subsets ─────────────────────────────────────────────────────
    for r in range(2, len(keys) + 1):
        for subset in itertools.combinations(keys, r):
            w = {k: 1.0 for k in subset}
            probs = weighted_probs(logits, w)
            name = "+".join(subset)
            rows.append((name, eval_probs(probs, labels)))

    # ── Weighted pairs/triples grid search ───────────────────────────────────────
    # For each pair/triple, sweep weight ratios [0.5, 1.0, 1.5, 2.0, 3.0]
    weight_values = [0.5, 1.0, 1.5, 2.0, 3.0]

    for r in range(2, min(len(keys), 4) + 1):
        for subset in itertools.combinations(keys, r):
            for combo in itertools.product(weight_values, repeat=r):
                if len(set(combo)) == 1:
                    continue  # equal weights already covered above
                w = dict(zip(subset, combo))
                probs = weighted_probs(logits, w)
                w_str = "+".join(f"{k}×{v}" for k, v in w.items())
                rows.append((w_str, eval_probs(probs, labels)))

    # ── Sort by best selection at 80% recall (index 1) and print ────────────────
    rows.sort(key=lambda x: x[1][1][0])  # sort by sel at 80% recall

    col_w = 18
    header = f"  {'Config':<45}" + "  ".join(f"{'R'+f'{t:.0%}':>{col_w}}" for t in TARGETS)
    print("\n" + "=" * (45 + col_w * len(TARGETS) + 4))
    print("  Sorted by min selection @ 80% recall")
    print("=" * (45 + col_w * len(TARGETS) + 4))
    print(header)
    print("  " + "-" * (43 + col_w * len(TARGETS) + 4))

    shown = set()
    for name, results in rows[:60]:
        # Deduplicate equivalent configs (equal-weight already in subset section)
        key = tuple(round(s, 3) for s, _ in results)
        if key in shown:
            continue
        shown.add(key)
        cells = "  ".join(f"{s:.1%}({r:.0%})" for s, r in results)
        print(f"  {name:<45}{cells}")

    # ── Per-target winners ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Best config per recall target (lowest selection rate):")
    print("=" * 60)
    for ti, target in enumerate(TARGETS):
        best_name, best_results = min(rows, key=lambda x: x[1][ti][0])
        sel, rec = best_results[ti]
        print(f"  R{target:.0%}  ->  {sel:.1%} sel ({rec:.0%} actual)   [{best_name}]")

    print()


if __name__ == "__main__":
    main()

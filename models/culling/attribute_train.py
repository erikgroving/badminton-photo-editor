"""
Train the attribute-aware culling model.

Usage:
    python -m models.culling.attribute_train
    python -m models.culling.attribute_train --backbone efficientnet_b3 --fn-weight 10

Checkpoint: checkpoints/culling_attr_<backbone>_w<fn_weight>.pt

After training, prints an attribute importance table showing which of the
8 computed features (blur, exposure, face quality, etc.) the model relied on.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    CHECKPOINTS_DIR, CULL_BATCH_SIZE, CULL_EPOCHS, CULL_FBETA,
    CULL_FN_WEIGHT, CULL_LR, MAPPING_FILE, THUMB_SIZE,
)
from data.attribute_extractor import ATTR_NAMES, build_attribute_cache
from data.mapping import flat_entries, load_mapping
from data.raw_reader import extract_thumbnail
from models.culling.attribute_model import AttributeAwareCullingModel
from models.culling.model import AsymmetricBCELoss
from models.culling.train import _evaluate_logits, _make_transforms, _print_metrics


def _ckpt_path(backbone: str, fn_weight: float) -> Path:
    safe = backbone.replace("/", "_")
    return CHECKPOINTS_DIR / f"culling_attr_{safe}_w{fn_weight:.0f}.pt"


# ── Dataset ───────────────────────────────────────────────────────────────────

class AttributeCullingDataset(torch.utils.data.Dataset):
    def __init__(self, entries: list[dict], attr_cache: dict, transform=None):
        self.entries     = entries
        self.attr_cache  = attr_cache
        self.transform   = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e   = self.entries[idx]
        img = extract_thumbnail(e["raw"], size=THUMB_SIZE)
        if self.transform:
            img = self.transform(img)

        attrs = torch.tensor(
            self.attr_cache.get(e["raw"], [0.0] * 8),
            dtype=torch.float32,
        )
        label = torch.tensor(float(e["label"]), dtype=torch.float32)
        return img, attrs, label


# ── Evaluation ────────────────────────────────────────────────────────────────

def _eval_attr(model, loader, device, fn_weight, fbeta):
    """Returns metrics dict using the same threshold sweep as the base model."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, attrs, labels in loader:
            logits = model(imgs.to(device), attrs.to(device)).squeeze(1)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return _evaluate_logits(logits, labels, fn_weight, fbeta)


# ── Training ──────────────────────────────────────────────────────────────────

def train(backbone: str, epochs: int, batch_size: int,
          fn_weight: float, lr: float) -> dict:

    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"Run data/mapping.py first.")

    mapping       = load_mapping()
    train_entries = [e for e in flat_entries(mapping, split="train")
                     if e["label"] is not None and e["raw"]]
    val_entries   = [e for e in flat_entries(mapping, split="val")
                     if e["label"] is not None and e["raw"]]
    test_entries  = [e for e in flat_entries(mapping, split="test")
                     if e["label"] is not None and e["raw"]]

    all_paths = [e["raw"] for e in train_entries + val_entries + test_entries]
    print(f"Pre-computing attributes for {len(all_paths):,} images...")
    attr_cache = build_attribute_cache(all_paths, max_workers=8)

    pos = sum(e["label"] for e in train_entries)
    neg = len(train_entries) - pos
    print(f"\n[attr/{backbone}] Train {len(train_entries):,}  kept={pos:,}  culled={neg:,}")

    sample_weights = [fn_weight if e["label"] == 1 else 1.0 for e in train_entries]
    sampler        = WeightedRandomSampler(sample_weights,
                                           num_samples=len(train_entries),
                                           replacement=True)

    tf_train = _make_transforms(train=True)
    tf_eval  = _make_transforms(train=False)

    train_ds = AttributeCullingDataset(train_entries, attr_cache, tf_train)
    val_ds   = AttributeCullingDataset(val_entries,   attr_cache, tf_eval)
    test_ds  = AttributeCullingDataset(test_entries,  attr_cache, tf_eval)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = AttributeAwareCullingModel(backbone=backbone).to(device)
    criterion = AsymmetricBCELoss(fn_weight=fn_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_path = _ckpt_path(backbone, fn_weight)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_cost    = float("inf")
    best_metrics: dict = {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, attrs, labels in tqdm(train_loader,
                                        desc=f"[attr/{backbone}] {epoch}/{epochs}",
                                        leave=False):
            imgs, attrs, labels = imgs.to(device), attrs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs, attrs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        val_m = _eval_attr(model, val_loader, device, fn_weight, CULL_FBETA)
        print(f"  loss={total_loss/len(train_loader):.4f}  [val]", end="")
        _print_metrics(val_m, CULL_FBETA)

        if val_m["asym_cost"] < best_cost:
            best_cost    = val_m["asym_cost"]
            best_metrics = dict(val_m, backbone=backbone, epoch=epoch,
                                fn_weight=fn_weight, model_type="attribute")
            torch.save({
                "epoch":       epoch,
                "backbone":    backbone,
                "fn_weight":   fn_weight,
                "model_state": model.state_dict(),
                "threshold":   val_m["threshold"],
                "metrics":     val_m,
            }, ckpt_path)
            print(f"    [OK] Saved best val (asym_cost={best_cost:.1f})")

    # Load best and evaluate on test set
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_m = _eval_attr(model, test_loader, device, fn_weight, CULL_FBETA)
    print(f"\n[attr/{backbone}] TEST SET:")
    _print_metrics(test_m, CULL_FBETA)
    best_metrics["test_metrics"] = test_m

    # Attribute importance table
    model.to("cpu")
    importance = model.attribute_importance()
    print(f"\n[attr/{backbone}] Attribute importance (what the model relied on):")
    print(f"  {'Attribute':<22}  {'Importance':>10}")
    print("  " + "-" * 34)
    for name, score in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "#" * int(score * 40)
        print(f"  {name:<22}  {score:>9.1%}  {bar}")
    best_metrics["attribute_importance"] = importance

    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",   type=str,   default="efficientnet_b0")
    parser.add_argument("--epochs",     type=int,   default=CULL_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=CULL_BATCH_SIZE)
    parser.add_argument("--fn-weight",  type=float, default=CULL_FN_WEIGHT)
    parser.add_argument("--lr",         type=float, default=CULL_LR)
    args = parser.parse_args()
    train(args.backbone, args.epochs, args.batch_size, args.fn_weight, args.lr)

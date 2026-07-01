"""
Train the culling classifier with a specified backbone.

Usage:
    python -m models.culling.train [--backbone NAME] [--epochs N] [--fn-weight F]

Checkpoint saved to checkpoints/culling_<backbone>.pt

Evaluation reports:
  - Asymmetric cost  (FN × fn_weight + FP)   — primary objective
  - Recall           (% of Jay's kept photos the model also keeps)
  - Selection rate   (% of all photos the model marks as "keep")
  - F-beta score     (recall-weighted, β=2)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

_LOGS_DIR = Path(__file__).parent.parent.parent / "logs"

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    CHECKPOINTS_DIR, CULL_BATCH_SIZE, CULL_EPOCHS, CULL_FBETA, CULL_FN_WEIGHT,
    CULL_LR, CULL_MODEL_NAME, MAPPING_FILE, THUMB_SIZE,
)
from data.mapping import flat_entries, load_mapping
from data.raw_reader import extract_thumbnail
from models.culling.model import AsymmetricBCELoss, build_model


def _ckpt_path(backbone: str, fn_weight: float | None = None, suffix: str = "") -> Path:
    safe = backbone.replace("/", "_")
    if fn_weight is not None and fn_weight != CULL_FN_WEIGHT:
        return CHECKPOINTS_DIR / f"culling_{safe}_w{fn_weight:.0f}{suffix}.pt"
    return CHECKPOINTS_DIR / f"culling_{safe}{suffix}.pt"


# ── Dataset ────────────────────────────────────────────────────────────────────

class CullingDataset(torch.utils.data.Dataset):
    def __init__(self, entries: list[dict], transform=None,
                 thumb_size: tuple[int, int] = THUMB_SIZE):
        self.entries = entries
        self.transform = transform
        self.thumb_size = thumb_size

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img = extract_thumbnail(e["raw"], size=self.thumb_size)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(e["label"], dtype=torch.float32)


def _make_transforms(train: bool, input_size: int = THUMB_SIZE,
                     mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    resize = ([transforms.Resize((input_size, input_size))]
              if input_size != THUMB_SIZE else [])
    norm = transforms.Normalize(list(mean), list(std))
    if train:
        return transforms.Compose(resize + [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose(resize + [
        transforms.ToTensor(),
        norm,
    ])


# ── Metrics ────────────────────────────────────────────────────────────────────

def fbeta(tp: int, fp: int, fn: int, beta: float) -> float:
    b2 = beta ** 2
    denom = (1 + b2) * tp + b2 * fn + fp
    return (1 + b2) * tp / denom if denom > 0 else 0.0


def _evaluate_logits(logits: torch.Tensor, labels: torch.Tensor,
                     fn_weight: float, beta: float) -> dict:
    """
    Shared threshold sweep given pre-computed logit and label tensors.
    Picks the threshold that minimises asymmetric cost (fn_weight×FN + FP).
    """
    probs = torch.sigmoid(logits)
    total = len(labels)

    best_cost = float("inf")
    best_t    = 0.5
    for t in torch.linspace(0.05, 0.95, 180):
        preds = (probs >= t).float()
        fn = int(((preds == 0) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        cost = fn_weight * fn + fp
        if cost < best_cost:
            best_cost, best_t = cost, float(t)

    preds = (probs >= best_t).float()
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    gt_kept    = tp + fn
    model_kept = tp + fp

    recall         = tp / gt_kept      if gt_kept    > 0 else 0.0
    selection_rate = model_kept / total if total      > 0 else 0.0
    precision      = tp / model_kept   if model_kept > 0 else 0.0
    fb             = fbeta(tp, fp, fn, beta)

    return {
        "threshold":      best_t,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall":          recall,
        "precision":       precision,
        "selection_rate":  selection_rate,
        f"f{beta:.0f}":    fb,
        "asym_cost":       fn_weight * fn + fp,
        "gt_kept":         gt_kept,
        "model_kept":      model_kept,
        "total":           total,
    }


def _evaluate(model, loader, device, fn_weight: float, beta: float) -> dict:
    """
    Collect logits from model+loader, then delegate to _evaluate_logits.
    """
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device)).cpu()
            all_logits.append(logits.squeeze(1))
            all_labels.append(labels)
    return _evaluate_logits(torch.cat(all_logits), torch.cat(all_labels),
                            fn_weight, beta)


def _print_metrics(m: dict, beta: float) -> None:
    beta_key = f"f{beta:.0f}"
    print(
        f"  recall={m['recall']:.3f} ({m['tp']}/{m['gt_kept']} of Jay's kept photos)"
        f"  selection={m['selection_rate']:.3f} ({m['model_kept']}/{m['total']} photos marked keep)"
        f"  {beta_key}={m[beta_key]:.4f}"
        f"  asym_cost={m['asym_cost']:.1f}  (FN×{int(m['asym_cost'] - m['fp'])/max(1,m['fn']):.0f} + FP)"
        f"  @thresh={m['threshold']:.3f}"
    )


# ── Training loop ──────────────────────────────────────────────────────────────

def train(backbone: str, epochs: int, batch_size: int, fn_weight: float, lr: float,
          mapping_file: Path | None = None, ckpt_suffix: str = "",
          grad_checkpoint: bool = False, warmup_epochs: int = 0,
          force_input_size: int | None = None,
          dynamic_img_size: bool = False) -> dict:
    mapping_file = mapping_file or MAPPING_FILE
    if not mapping_file.exists():
        raise FileNotFoundError(f"Run data/mapping.py first — {mapping_file} not found.")

    with open(mapping_file) as fh:
        mapping = json.load(fh)
    train_entries = [e for e in flat_entries(mapping, split="train") if e["label"] is not None and e["raw"]]
    val_entries   = [e for e in flat_entries(mapping, split="val")   if e["label"] is not None and e["raw"]]
    test_entries  = [e for e in flat_entries(mapping, split="test")  if e["label"] is not None and e["raw"]]

    pos = sum(e["label"] for e in train_entries)
    neg = len(train_entries) - pos
    print(f"[{backbone}] Train: {len(train_entries):,}  (kept={pos:,}  rejected={neg:,})")
    print(f"[{backbone}] Val:   {len(val_entries):,}  |  Test: {len(test_entries):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{backbone}] Device: {device}")

    model = build_model(backbone=backbone, pretrained=True,
                        dynamic_img_size=dynamic_img_size).to(device)

    # Resolve normalization stats from timm; optionally override input size
    import timm
    data_cfg  = timm.data.resolve_model_data_config(model)
    norm_mean = data_cfg["mean"]
    norm_std  = data_cfg["std"]
    if force_input_size is not None:
        input_size = force_input_size
        print(f"[{backbone}] input_size overridden to {input_size} (dynamic_img_size={dynamic_img_size})")
    else:
        input_size = data_cfg["input_size"][1]  # (C, H, W) -> H
    if input_size != THUMB_SIZE:
        print(f"[{backbone}] Resizing input {THUMB_SIZE} -> {input_size}")
    if norm_mean != (0.485, 0.456, 0.406):
        print(f"[{backbone}] Non-ImageNet normalisation: mean={norm_mean}")

    # Read thumbnails at model input size (avoids upscaling from a smaller thumb)
    thumb_size = (input_size, input_size)

    sample_weights = [fn_weight if e["label"] == 1 else 1.0 for e in train_entries]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_entries), replacement=True)

    tf_args = dict(input_size=input_size, mean=norm_mean, std=norm_std)
    train_ds    = CullingDataset(train_entries, _make_transforms(True,  **tf_args), thumb_size=thumb_size)
    val_ds      = CullingDataset(val_entries,   _make_transforms(False, **tf_args), thumb_size=thumb_size)
    test_ds     = CullingDataset(test_entries,  _make_transforms(False, **tf_args), thumb_size=thumb_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,   num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,   num_workers=4, pin_memory=True)

    if grad_checkpoint:
        try:
            model.set_grad_checkpointing(enable=True)
            print(f"[{backbone}] Gradient checkpointing enabled")
        except AttributeError:
            print(f"[{backbone}] Gradient checkpointing not supported, skipping")

    criterion = AsymmetricBCELoss(fn_weight=fn_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if warmup_epochs > 0 and warmup_epochs < epochs:
        # Linear warmup from lr/100 → lr, then cosine decay for remaining epochs
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        print(f"[{backbone}] LR schedule: {warmup_epochs}-epoch warmup then cosine (lr={lr})")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_path = _ckpt_path(backbone, fn_weight, ckpt_suffix)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    _LOGS_DIR.mkdir(exist_ok=True)
    progress_file = _LOGS_DIR / f"progress_culling_{backbone.replace('/', '_')}.jsonl"
    progress_file.unlink(missing_ok=True)  # fresh file each run

    best_cost     = float("inf")
    best_metrics: dict = {}
    train_start   = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"[{backbone}] Epoch {epoch}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        epoch_s = time.time() - epoch_start

        # Use val to tune threshold and drive early stopping
        val_metrics = _evaluate(model, val_loader, device, fn_weight, CULL_FBETA)
        avg_loss = total_loss / len(train_loader)
        print(f"  loss={avg_loss:.4f}  [val]", end="")
        _print_metrics(val_metrics, CULL_FBETA)

        # Write per-epoch progress for watch_training.py
        with open(progress_file, "a", encoding="utf-8") as pf:
            pf.write(json.dumps({
                "ts":       time.time(),
                "backbone": backbone,
                "epoch":    epoch,
                "total":    epochs,
                "loss":     round(avg_loss, 5),
                "epoch_s":  round(epoch_s, 1),
                "elapsed_s": round(time.time() - train_start, 1),
                **{k: round(v, 5) if isinstance(v, float) else v
                   for k, v in val_metrics.items()
                   if k not in ("tp", "fp", "fn", "tn", "gt_kept", "model_kept", "total")},
            }) + "\n")

        if val_metrics["asym_cost"] < best_cost:
            best_cost    = val_metrics["asym_cost"]
            best_metrics = dict(val_metrics, backbone=backbone, epoch=epoch)
            torch.save({
                "epoch":            epoch,
                "backbone":         backbone,
                "model_state":      model.state_dict(),
                "threshold":        val_metrics["threshold"],
                "fn_weight":        fn_weight,
                "metrics":          val_metrics,
                "input_size":       input_size,
                "dynamic_img_size": dynamic_img_size,
                "norm_mean":        norm_mean,
                "norm_std":         norm_std,
            }, ckpt_path)
            print(f"    [OK] Saved best val (asym_cost={best_cost:.1f})")

    # Final evaluation on held-out test set using the best saved threshold
    ckpt       = torch.load(ckpt_path, map_location="cpu")
    best_thresh = ckpt["threshold"]
    model.load_state_dict(ckpt["model_state"])
    test_metrics = _evaluate(model, test_loader, device, fn_weight, CULL_FBETA)
    print(f"\n[{backbone}] TEST SET RESULTS:")
    _print_metrics(test_metrics, CULL_FBETA)
    best_metrics["test_metrics"] = test_metrics

    print(f"[{backbone}] Done. Best checkpoint: {ckpt_path}")
    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",    type=str,   default=CULL_MODEL_NAME)
    parser.add_argument("--epochs",      type=int,   default=CULL_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=CULL_BATCH_SIZE)
    parser.add_argument("--fn-weight",   type=float, default=CULL_FN_WEIGHT)
    parser.add_argument("--lr",          type=float, default=CULL_LR)
    parser.add_argument("--mapping",     type=Path,  default=None,
                        help="Path to an alternative mapping JSON (default: data/mapping.json)")
    parser.add_argument("--ckpt-suffix",      type=str,   default="",
                        help="Suffix appended to checkpoint name, e.g. '_event' -> culling_b0_event.pt")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing (saves VRAM for large ViTs)")
    parser.add_argument("--warmup-epochs",  type=int, default=0,
                        help="Linear LR warmup epochs before cosine decay (default: 0)")
    args = parser.parse_args()
    train(args.backbone, args.epochs, args.batch_size, args.fn_weight, args.lr,
          mapping_file=args.mapping, ckpt_suffix=args.ckpt_suffix,
          grad_checkpoint=args.grad_checkpoint, warmup_epochs=args.warmup_epochs)

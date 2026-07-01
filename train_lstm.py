"""
Train BurstLSTM on top of frozen EfficientNet-B3 embeddings.

Pre-requisite: run extract_b3_embeddings.py first to generate
  embeddings/b3_embeddings.pt and embeddings/burst_sequences.json

Checkpoint saved to: checkpoints/culling_lstm_b3.pt

Usage:
    python train_lstm.py
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINTS_DIR, CULL_FBETA, CULL_FN_WEIGHT
from data.sequence_dataset import BurstSequenceDataset, collate_sequences
from models.culling.lstm_model import BurstLSTM
from models.culling.train import fbeta

EPOCHS      = 25
LR          = 1e-3
BATCH_SIZE  = 32   # bursts per batch
FN_WEIGHT   = CULL_FN_WEIGHT
HIDDEN_SIZE = 256
NUM_LAYERS  = 2
DROPOUT     = 0.3
CKPT_PATH   = CHECKPOINTS_DIR / "culling_lstm_b3.pt"
LOG_PATH    = Path("logs/lstm_b3.log")


def masked_asym_bce(logits: torch.Tensor, labels: torch.Tensor,
                    lengths: torch.Tensor, fn_weight: float) -> torch.Tensor:
    batch, max_len = logits.shape
    mask = (torch.arange(max_len, device=logits.device).unsqueeze(0)
            < lengths.to(logits.device).unsqueeze(1))
    flat_logits = logits[mask]
    flat_labels = labels.to(logits.device)[mask]
    pw = torch.tensor([fn_weight], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(flat_logits, flat_labels, pos_weight=pw)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             fn_weight: float, beta: float) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for embs, labels, lengths in loader:
            logits = model(embs.to(device), lengths)
            # Unpack valid (non-padded) positions
            for i, L in enumerate(lengths):
                all_logits.append(logits[i, :L].cpu())
                all_labels.append(labels[i, :L])

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    probs = torch.sigmoid(logits_cat)
    total = len(labels_cat)

    best_cost, best_t = float("inf"), 0.5
    for t in torch.linspace(0.05, 0.95, 180):
        preds = (probs >= t).float()
        fn = int(((preds == 0) & (labels_cat == 1)).sum())
        fp = int(((preds == 1) & (labels_cat == 0)).sum())
        cost = fn_weight * fn + fp
        if cost < best_cost:
            best_cost, best_t = cost, float(t)

    preds = (probs >= best_t).float()
    tp = int(((preds == 1) & (labels_cat == 1)).sum())
    fp = int(((preds == 1) & (labels_cat == 0)).sum())
    fn = int(((preds == 0) & (labels_cat == 1)).sum())
    gt_kept    = tp + fn
    model_kept = tp + fp
    recall     = tp / gt_kept      if gt_kept    > 0 else 0.0
    sel        = model_kept / total if total      > 0 else 0.0
    fb         = fbeta(tp, fp, fn, beta)
    return {
        "threshold": best_t, "asym_cost": best_cost,
        "recall": recall, "selection_rate": sel,
        f"f{beta:.0f}": fb,
        "tp": tp, "fp": fp, "fn": fn,
        "gt_kept": gt_kept, "model_kept": model_kept, "total": total,
    }


def main():
    emb_file = Path("embeddings/b3_embeddings.pt")
    seq_file = Path("embeddings/burst_sequences.json")
    if not emb_file.exists() or not seq_file.exists():
        print("ERROR: run extract_b3_embeddings.py first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = BurstSequenceDataset("train", emb_file, seq_file)
    val_ds   = BurstSequenceDataset("val",   emb_file, seq_file)
    test_ds  = BurstSequenceDataset("test",  emb_file, seq_file)
    print(f"Bursts — train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    # Peek at embedding dim
    sample_emb, _, _ = collate_sequences([train_ds[0]])
    emb_dim = sample_emb.shape[-1]
    print(f"Embedding dim: {emb_dim}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_sequences, num_workers=0)
    val_loader   = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_sequences, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_sequences, num_workers=0)

    model = BurstLSTM(emb_dim=emb_dim, hidden_size=HIDDEN_SIZE,
                      num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    CKPT_PATH.parent.mkdir(exist_ok=True)
    LOG_PATH.parent.mkdir(exist_ok=True)
    LOG_PATH.unlink(missing_ok=True)

    best_cost     = float("inf")
    best_metrics: dict = {}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for embs, labels, lengths in tqdm(train_loader,
                                          desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            optimizer.zero_grad()
            logits = model(embs.to(device), lengths)
            loss   = masked_asym_bce(logits, labels, lengths, FN_WEIGHT)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        val_m = evaluate(model, val_loader, device, FN_WEIGHT, CULL_FBETA)
        avg_loss = total_loss / len(train_loader)
        print(f"  ep{epoch:02d}  loss={avg_loss:.4f}  [val]"
              f"  recall={val_m['recall']:.3f} ({val_m['tp']}/{val_m['gt_kept']})"
              f"  sel={val_m['selection_rate']:.3f} ({val_m['model_kept']}/{val_m['total']})"
              f"  cost={val_m['asym_cost']:.0f}"
              f"  @t={val_m['threshold']:.3f}")

        with open(LOG_PATH, "a") as fh:
            fh.write(json.dumps({
                "epoch": epoch, "loss": round(avg_loss, 5),
                **{k: round(v, 5) if isinstance(v, float) else v
                   for k, v in val_m.items()
                   if k not in ("tp", "fp", "fn", "gt_kept", "model_kept", "total")},
            }) + "\n")

        if val_m["asym_cost"] < best_cost:
            best_cost    = val_m["asym_cost"]
            best_metrics = dict(val_m)
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "threshold": val_m["threshold"], "fn_weight": FN_WEIGHT,
                "metrics": val_m,
                "emb_dim": emb_dim, "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
            }, CKPT_PATH)
            print(f"    [OK] saved best (cost={best_cost:.0f})")

    # Test set evaluation
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(model, test_loader, device, FN_WEIGHT, CULL_FBETA)
    print(f"\nTEST SET:"
          f"  recall={test_m['recall']:.3f} ({test_m['tp']}/{test_m['gt_kept']})"
          f"  sel={test_m['selection_rate']:.3f}"
          f"  cost={test_m['asym_cost']:.0f}"
          f"  @t={test_m['threshold']:.3f}")
    print(f"Done. Checkpoint: {CKPT_PATH}")


if __name__ == "__main__":
    main()

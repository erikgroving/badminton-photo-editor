"""
Train the judge discriminator with a specified backbone.

Positives ("edited", label=1): all edited JPGs + Boba Cup judge-only edits.
Negatives ("raw",    label=0): embedded thumbnails from Raws (balanced).

Usage:
    python -m models.judge.train [--backbone NAME] [--epochs N]
    python -m models.judge.sweep                  # compare all backbones
"""
import argparse
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    CHECKPOINTS_DIR, JUDGE_BATCH_SIZE, JUDGE_EPOCHS, JUDGE_LR,
    JUDGE_MODEL_NAME, MAPPING_FILE, THUMB_SIZE,
)
from data.mapping import flat_entries, load_mapping
from data.raw_reader import extract_thumbnail, mask_watermark
from models.judge.model import build_model


def _ckpt_path(backbone: str) -> Path:
    safe = backbone.replace("/", "_")
    return CHECKPOINTS_DIR / f"judge_{safe}.pt"


# ── Dataset ────────────────────────────────────────────────────────────────────

class JudgeDataset(torch.utils.data.Dataset):
    def __init__(self, entries: list[dict], transform=None):
        self.entries  = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        from PIL import Image
        e = self.entries[idx]
        if e["type"] == "edited":
            img = Image.open(e["path"]).convert("RGB")
            img.thumbnail(THUMB_SIZE)
            canvas = Image.new("RGB", THUMB_SIZE, (0, 0, 0))
            canvas.paste(img, ((THUMB_SIZE[0] - img.width)  // 2,
                               (THUMB_SIZE[1] - img.height) // 2))
            img = mask_watermark(canvas)   # remove watermark so judge can't cheat
        else:
            img = mask_watermark(extract_thumbnail(e["path"], size=THUMB_SIZE))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(e["label"], dtype=torch.float32)


def build_entries(mapping: dict, split: str) -> list[dict]:
    """
    Build judge entries for a specific split (train/val/test).
    Positives: edited photos from that split + judge_only edits (always train).
    Negatives: raw thumbnails from that split, balanced to match positives.
    """
    positives = []
    for e in flat_entries(mapping, split=split):
        if e["edited"] and e["label"] == 1:
            positives.append({"path": e["edited"], "label": 1, "type": "edited"})
    if split == "train":
        # Include judge_only edits (Boba Cup etc.) in training positives
        for e in flat_entries(mapping, split="judge_only"):
            if e["edited"]:
                positives.append({"path": e["edited"], "label": 1, "type": "edited"})

    all_raws  = [e for e in flat_entries(mapping, split=split) if e["raw"] and e["label"] is not None]
    rng       = random.Random(42)
    negatives = rng.sample(all_raws, min(len(positives), len(all_raws)))
    negatives = [{"path": e["raw"], "label": 0, "type": "raw"} for e in negatives]
    entries   = positives + negatives
    rng.shuffle(entries)
    return entries


# ── Eval ───────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device) -> dict:
    model.eval()
    correct = total = tp = fp = tn = fn = 0
    with torch.no_grad():
        for imgs, labels in loader:
            preds = (torch.sigmoid(model(imgs.to(device)).squeeze(1).cpu()) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total   += len(labels)
            tp += int(((preds == 1) & (labels == 1)).sum())
            fp += int(((preds == 1) & (labels == 0)).sum())
            tn += int(((preds == 0) & (labels == 0)).sum())
            fn += int(((preds == 0) & (labels == 1)).sum())
    return {"accuracy": correct / total, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ── Training loop ──────────────────────────────────────────────────────────────

def train(backbone: str, epochs: int, batch_size: int, lr: float) -> dict:
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"Run data/mapping.py first — {MAPPING_FILE} not found.")

    mapping       = load_mapping()
    train_entries = build_entries(mapping, "train")
    val_entries   = build_entries(mapping, "val")
    test_entries  = build_entries(mapping, "test")
    print(f"[{backbone}] Train: {len(train_entries):,}  Val: {len(val_entries):,}  Test: {len(test_entries):,}")

    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(JudgeDataset(train_entries, tf_train), batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(JudgeDataset(val_entries,   tf_val),   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(JudgeDataset(test_entries,  tf_val),   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = build_model(backbone=backbone, pretrained=True).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt = _ckpt_path(backbone)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_metrics: dict = {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"[{backbone}] Epoch {epoch}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs).squeeze(1), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        m = evaluate(model, val_loader, device)
        print(f"  [{backbone}] loss={total_loss/len(train_loader):.4f}  acc={m['accuracy']:.4f}"
              f"  TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")

        if m["accuracy"] > best_acc:
            best_acc     = m["accuracy"]
            best_metrics = dict(m, backbone=backbone, epoch=epoch)
            torch.save({"epoch": epoch, "backbone": backbone,
                        "model_state": model.state_dict(), "metrics": m}, ckpt)
            print(f"    >> Saved best (val_acc={best_acc:.4f})")

    # Final evaluation on held-out test set
    test_m = evaluate(model, test_loader, device)
    print(f"[{backbone}] Final test accuracy: {test_m['accuracy']:.4f}  TP={test_m['tp']} FP={test_m['fp']} TN={test_m['tn']} FN={test_m['fn']}")
    best_metrics["test_accuracy"] = test_m["accuracy"]
    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",   type=str,   default=JUDGE_MODEL_NAME)
    parser.add_argument("--epochs",     type=int,   default=JUDGE_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=JUDGE_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=JUDGE_LR)
    args = parser.parse_args()
    train(args.backbone, args.epochs, args.batch_size, args.lr)

"""
For each trained backbone, sweep inference thresholds to find the
selection rate required to achieve target recall levels.

Usage:
    python recall_vs_selection.py
"""
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
from data.raw_reader import extract_thumbnail
from models.culling.model import build_model
from models.culling.train import CullingDataset

TARGETS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98]

BACKBONES = [
    "efficientnet_b0",
    "efficientnet_b3",
    "resnet50",
    "mobilenetv3_large_100",
]

# Extra named checkpoints to show alongside the standard backbone list.
# Each entry: (display_label, checkpoint_path_str, backbone_name, mapping_file_or_None)
# When mapping_file is None, uses the default mapping.json (burst-1s val set).
# Specifying a mapping_file means the model is evaluated on its own val set — fair
# within-split numbers, but not directly comparable across rows with different mappings.
EXTRA_CHECKPOINTS: list[tuple] = [
    # Split-strategy comparison (each on its own val set)
    ("mobilenetv3 [event]",    "checkpoints/culling_mobilenetv3_large_100_event.pt",    "mobilenetv3_large_100", "data/mapping_event.json"),
    ("mobilenetv3 [burst-60]", "checkpoints/culling_mobilenetv3_large_100_burst60.pt",  "mobilenetv3_large_100", "data/mapping_burst60.json"),
    ("mobilenetv3 [burst-1s]", "checkpoints/culling_mobilenetv3_large_100.pt",          "mobilenetv3_large_100", None),
    ("mobilenetv3 [random]",   "checkpoints/culling_mobilenetv3_large_100_random.pt",   "mobilenetv3_large_100", "data/mapping_random.json"),
    # Large / modern-pretrained models (burst-1s val)
    ("siglip_512 [base]",      "checkpoints/culling_vit_base_patch16_siglip_512.pt",    "vit_base_patch16_siglip_512",  None),
    ("dinov2 [base+reg]",      "checkpoints/culling_vit_base_patch14_reg4_dinov2.pt",   "vit_base_patch14_reg4_dinov2", None),
    ("dinov2 [large]",         "checkpoints/culling_vit_large_patch14_dinov2.pt",       "vit_large_patch14_dinov2",     None),
]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _build_val_transform(input_size: int, mean: tuple, std: tuple):
    resize = ([transforms.Resize((input_size, input_size))]
              if input_size != THUMB_SIZE else [])
    return transforms.Compose(resize + [
        transforms.ToTensor(),
        transforms.Normalize(list(mean), list(std)),
    ])


def get_logits(backbone: str, val_entries: list, device: torch.device,
               ckpt_path: str | None = None) -> tuple:
    if ckpt_path is None:
        ckpt_path = str(CHECKPOINTS_DIR / f"culling_{backbone.replace('/', '_')}.pt")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = build_model(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    # Old checkpoints lack input_size/norm — they were all trained at THUMB_SIZE with ImageNet norm
    input_size = ck.get("input_size", THUMB_SIZE)
    norm_mean  = ck.get("norm_mean",  _IMAGENET_MEAN)
    norm_std   = ck.get("norm_std",   _IMAGENET_STD)
    tf = _build_val_transform(input_size, norm_mean, norm_std)

    ds     = CullingDataset(val_entries, tf)
    loader = DataLoader(ds, batch_size=CULL_BATCH_SIZE * 2,
                        shuffle=False, num_workers=4, pin_memory=True)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            all_logits.append(model(imgs.to(device)).squeeze(1).cpu())
            all_labels.append(labels)

    return torch.cat(all_logits), torch.cat(all_labels)


def threshold_for_recall(probs, labels, target_recall: float) -> float:
    """Sweep 500 thresholds; return the highest one achieving >= target recall
    (highest threshold = smallest selection rate that still hits the target)."""
    best_t = 0.0
    for t in torch.linspace(0.001, 0.999, 500):
        t = float(t)
        preds  = (probs >= t).float()
        tp     = int(((preds == 1) & (labels == 1)).sum())
        fn     = int(((preds == 0) & (labels == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if recall >= target_recall:
            best_t = t   # keep updating — want the highest t that still meets target
    return best_t


def selection_at_threshold(probs, labels, t: float) -> float:
    preds = (probs >= t).float()
    return float(preds.sum()) / len(preds)


def get_ensemble_probs(val_entries: list, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Average sigmoid probabilities from all four clean burst-1s backbones."""
    prob_sum = None
    labels_out = None
    for backbone in BACKBONES:
        ckpt_path = str(CHECKPOINTS_DIR / f"culling_{backbone.replace('/', '_')}.pt")
        if not Path(ckpt_path).exists():
            continue
        logits, labels = get_logits(backbone, val_entries, device, ckpt_path=ckpt_path)
        p = torch.sigmoid(logits)
        prob_sum = p if prob_sum is None else prob_sum + p
        labels_out = labels
    return (prob_sum / len(BACKBONES)), labels_out


def get_named_ensemble_probs(members: list[tuple[str, str]], val_entries: list,
                             device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Average sigmoid probs for an arbitrary list of (backbone, ckpt_path) pairs."""
    prob_sum = None
    labels_out = None
    n = 0
    for backbone, ckpt_path in members:
        if not Path(ckpt_path).exists():
            continue
        logits, labels = get_logits(backbone, val_entries, device, ckpt_path=ckpt_path)
        p = torch.sigmoid(logits)
        prob_sum = p if prob_sum is None else prob_sum + p
        labels_out = labels
        n += 1
    return (prob_sum / n), labels_out


# Named ensemble configs — (label, [(backbone, ckpt_path), ...])
_LARGE_MEMBERS = [
    ("vit_base_patch16_siglip_512",  str(CHECKPOINTS_DIR / "culling_vit_base_patch16_siglip_512.pt")),
    ("vit_base_patch14_reg4_dinov2", str(CHECKPOINTS_DIR / "culling_vit_base_patch14_reg4_dinov2.pt")),
    ("vit_large_patch14_dinov2",     str(CHECKPOINTS_DIR / "culling_vit_large_patch14_dinov2.pt")),
]
_SMALL_MEMBERS = [
    (bb, str(CHECKPOINTS_DIR / f"culling_{bb.replace('/', '_')}.pt"))
    for bb in BACKBONES
]

NAMED_ENSEMBLES: list[tuple[str, list]] = [
    ("ensemble: 3 large models",          _LARGE_MEMBERS),
    ("ensemble: small+siglip",            _SMALL_MEMBERS + [_LARGE_MEMBERS[0]]),
    ("ensemble: small+dinov2_base",       _SMALL_MEMBERS + [_LARGE_MEMBERS[1]]),
    ("ensemble: small+dinov2_large",      _SMALL_MEMBERS + [_LARGE_MEMBERS[2]]),
    ("ensemble: small+all large",         _SMALL_MEMBERS + _LARGE_MEMBERS),
    ("ensemble: dinov2_base+large",       [_LARGE_MEMBERS[1], _LARGE_MEMBERS[2]]),
]


def _load_val_entries(mapping_path: str | None) -> list:
    if mapping_path is None:
        mapping = load_mapping()
    else:
        with open(mapping_path) as fh:
            mapping = json.load(fh)
    return [e for e in flat_entries(mapping, split="val")
            if e["label"] is not None and e["raw"]]


def main():
    default_val = _load_val_entries(None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDefault val set (burst-1s): {len(default_val):,} images  |  device: {device}")
    print(f"Targets: {[f'{t:.0%}' for t in TARGETS]} recall\n")

    col = 10
    header = f"  {'Backbone':<38}  {'Val set':<10}" + "".join(f"  {'R'+f'{t:.0%}':>{col}}" for t in TARGETS)
    print(header)
    print("  " + "-" * (38 + 12 + (col + 2) * len(TARGETS)))

    # Standard backbone rows — all use default (burst-1s) val set
    rows: list[tuple] = []
    for backbone in BACKBONES:
        pt = CHECKPOINTS_DIR / f"culling_{backbone.replace('/', '_')}.pt"
        rows.append((backbone, str(pt) if pt.exists() else None, backbone, None))

    # Extra rows from EXTRA_CHECKPOINTS (4-tuple: label, ckpt, backbone, mapping)
    for entry in EXTRA_CHECKPOINTS:
        label, ckpt_path, backbone, mapping_file = entry
        rows.append((label, ckpt_path if Path(ckpt_path).exists() else None, backbone, mapping_file))

    val_cache: dict[str | None, list] = {None: default_val}

    for display, ckpt_path, backbone, mapping_file in rows:
        if ckpt_path is None or not Path(ckpt_path).exists():
            print(f"  {display:<38}  {'—':<10}  (no checkpoint)")
            continue

        if mapping_file not in val_cache:
            val_cache[mapping_file] = _load_val_entries(mapping_file)
        val_entries = val_cache[mapping_file]
        val_label   = Path(mapping_file).stem.replace("mapping_", "") if mapping_file else "burst-1s"

        logits, labels = get_logits(backbone, val_entries, device, ckpt_path=ckpt_path)
        probs = torch.sigmoid(logits)

        cells = []
        for target in TARGETS:
            t      = threshold_for_recall(probs, labels, target)
            sel    = selection_at_threshold(probs, labels, t)
            preds  = (probs >= t).float()
            tp     = int(((preds == 1) & (labels == 1)).sum())
            fn     = int(((preds == 0) & (labels == 1)).sum())
            actual = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            cells.append(f"{sel:.1%} ({actual:.0%})")

        row = f"  {display:<38}  {val_label:<10}" + "".join(f"  {c:>{col}}" for c in cells)
        print(row)

    # Standard small-model ensemble
    all_present = all(
        (CHECKPOINTS_DIR / f"culling_{bb.replace('/', '_')}.pt").exists()
        for bb in BACKBONES
    )
    if all_present:
        print(f"  {'--- ensemble ---':<38}  {'burst-1s':<10}")
        ens_probs, ens_labels = get_ensemble_probs(default_val, device)
        cells = []
        for target in TARGETS:
            t      = threshold_for_recall(ens_probs, ens_labels, target)
            sel    = selection_at_threshold(ens_probs, ens_labels, t)
            preds  = (ens_probs >= t).float()
            tp     = int(((preds == 1) & (ens_labels == 1)).sum())
            fn     = int(((preds == 0) & (ens_labels == 1)).sum())
            actual = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            cells.append(f"{sel:.1%} ({actual:.0%})")
        print(f"  {'ensemble (b0+b3+r50+mv3)':<38}  {'burst-1s':<10}" +
              "".join(f"  {c:>{col}}" for c in cells))

    # Named ensembles including large models
    for ens_label, members in NAMED_ENSEMBLES:
        available = [(bb, cp) for bb, cp in members if Path(cp).exists()]
        if len(available) < 2:
            continue
        ens_probs, ens_labels = get_named_ensemble_probs(available, default_val, device)
        cells = []
        for target in TARGETS:
            t      = threshold_for_recall(ens_probs, ens_labels, target)
            sel    = selection_at_threshold(ens_probs, ens_labels, t)
            preds  = (ens_probs >= t).float()
            tp     = int(((preds == 1) & (ens_labels == 1)).sum())
            fn     = int(((preds == 0) & (ens_labels == 1)).sum())
            actual = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            cells.append(f"{sel:.1%} ({actual:.0%})")
        print(f"  {ens_label:<38}  {'burst-1s':<10}" +
              "".join(f"  {c:>{col}}" for c in cells))

    # LSTM row (if checkpoint + embeddings exist)
    lstm_ckpt = CHECKPOINTS_DIR / "culling_lstm_b3.pt"
    emb_file  = Path("embeddings/b3_embeddings.pt")
    seq_file  = Path("embeddings/burst_sequences.json")
    if lstm_ckpt.exists() and emb_file.exists() and seq_file.exists():
        from models.culling.lstm_model import BurstLSTM
        from data.sequence_dataset import BurstSequenceDataset, collate_sequences
        from torch.utils.data import DataLoader as _DL
        import json as _json

        ck = torch.load(str(lstm_ckpt), map_location=device, weights_only=False)
        lstm = BurstLSTM(emb_dim=ck["emb_dim"], hidden_size=ck["hidden_size"],
                         num_layers=ck["num_layers"]).to(device)
        lstm.load_state_dict(ck["model_state"])
        lstm.eval()

        val_ds = BurstSequenceDataset("val", emb_file, seq_file)
        val_loader = _DL(val_ds, batch_size=32, shuffle=False,
                         collate_fn=collate_sequences, num_workers=0)

        all_logits, all_labels = [], []
        with torch.no_grad():
            for embs, labels, lengths in val_loader:
                logits = lstm(embs.to(device), lengths)
                for i, L in enumerate(lengths):
                    all_logits.append(logits[i, :L].cpu())
                    all_labels.append(labels[i, :L])
        lstm_probs  = torch.sigmoid(torch.cat(all_logits))
        lstm_labels = torch.cat(all_labels)

        cells = []
        for target in TARGETS:
            t      = threshold_for_recall(lstm_probs, lstm_labels, target)
            sel    = selection_at_threshold(lstm_probs, lstm_labels, t)
            preds  = (lstm_probs >= t).float()
            tp     = int(((preds == 1) & (lstm_labels == 1)).sum())
            fn     = int(((preds == 0) & (lstm_labels == 1)).sum())
            actual = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            cells.append(f"{sel:.1%} ({actual:.0%})")
        print(f"  {'b3+lstm [burst-1s]':<38}  {'burst-1s':<10}" +
              "".join(f"  {c:>{col}}" for c in cells))

    print()
    print("  Format: selection_rate (actual_recall achieved)")
    print("  Val set column: which mapping's val split was used for evaluation")
    print("  Lower selection = fewer photos Jay needs to review")


if __name__ == "__main__":
    main()

"""
Unsupervised burst-cluster culling baseline.

Strategy:
  1. Extract embeddings for every val photo using a pretrained backbone (no fine-tuning).
  2. Within each burst group (EXIF-timestamp proximity, 1-second window), select the K
     photos whose embeddings are closest to the burst centroid.
  3. Evaluate: what recall and selection rate does this achieve vs Jay's labels?

Usage:
    python cluster_culling.py [--top-k K]
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import CULL_BATCH_SIZE, THUMB_SIZE
from data.mapping import (
    _build_ts_raw_map, _group_into_bursts, _ts_str_to_float, flat_entries, load_mapping,
)
from data.raw_reader import extract_thumbnail
from models.culling.model import build_model

_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EMBEDDING_BACKBONE = "efficientnet_b0"
_NUM_RE = re.compile(r"0V2A(\d{4,})", re.IGNORECASE)


class EmbedDataset(Dataset):
    def __init__(self, entries: list[dict]):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img = extract_thumbnail(e["raw"], size=THUMB_SIZE)
        return _TF(img), idx


def extract_embeddings(entries: list[dict], device: torch.device) -> torch.Tensor:
    """Return (N, D) L2-normalised embedding matrix using pretrained backbone features."""
    model = build_model(backbone=EMBEDDING_BACKBONE, pretrained=True).to(device)
    model.classifier = torch.nn.Identity()
    model.eval()

    ds = EmbedDataset(entries)
    loader = DataLoader(ds, batch_size=CULL_BATCH_SIZE * 2, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_embs: list[torch.Tensor] = [None] * len(entries)
    with torch.no_grad():
        for imgs, idxs in tqdm(loader, desc="Extracting embeddings"):
            feats = model(imgs.to(device)).cpu()
            for feat, i in zip(feats, idxs.tolist()):
                all_embs[i] = feat

    embs = torch.stack(all_embs)
    return F.normalize(embs, dim=1)


def build_burst_groups(entries: list[dict]) -> list[list[int]]:
    """
    Re-apply burst-1s grouping to a flat entry list.
    Groups by parent directory first, then by EXIF timestamp proximity (1-second window).
    Returns list of groups; each group is a list of indices into `entries`.
    """
    path_to_idx = {e["raw"]: i for i, e in enumerate(entries)}

    # Collect raw files per directory
    by_dir: dict[Path, list[Path]] = defaultdict(list)
    for e in entries:
        by_dir[Path(e["raw"]).parent].append(Path(e["raw"]))

    groups: list[list[int]] = []
    for dir_path, raw_files in by_dir.items():
        # _build_ts_raw_map wants a directory; returns {ts_str: raw_path}
        ts_raw = _build_ts_raw_map(dir_path)
        raw_timestamps: dict[str, float] = {
            str(raw_f): ts_f
            for ts_str, raw_f in ts_raw.items()
            if (ts_f := _ts_str_to_float(ts_str)) is not None
        }
        for burst in _group_into_bursts(raw_files, raw_timestamps):
            idxs = [path_to_idx[str(f)] for f in burst if str(f) in path_to_idx]
            if idxs:
                groups.append(idxs)

    return groups


def score_top_k(embs: torch.Tensor, labels: torch.Tensor,
                groups: list[list[int]], k: int) -> tuple[float, float]:
    """Select top-k closest-to-centroid per group; return (recall, selection_rate)."""
    selected = torch.zeros(len(labels))
    for idxs in groups:
        if len(idxs) <= k:
            for i in idxs:
                selected[i] = 1.0
        else:
            g = embs[idxs]
            centroid = g.mean(0, keepdim=True)
            sims = (g * centroid).sum(1)
            _, top_i = sims.topk(k)
            for i in top_i.tolist():
                selected[idxs[i]] = 1.0

    tp = int(((selected == 1) & (labels == 1)).sum())
    fp = int(((selected == 1) & (labels == 0)).sum())
    fn = int(((selected == 0) & (labels == 1)).sum())
    recall   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sel_rate = (tp + fp) / len(labels)
    return recall, sel_rate


def evaluate_cluster_selection(top_k: int) -> None:
    mapping = load_mapping()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_entries = [e for e in flat_entries(mapping, split="val")
                   if e["label"] is not None and e["raw"]]
    labels = torch.tensor([e["label"] for e in val_entries], dtype=torch.float32)

    print(f"\nBuilding burst groups from val set ({len(val_entries):,} images)...")
    groups = build_burst_groups(val_entries)
    sizes  = [len(g) for g in groups]
    print(f"  {len(groups):,} burst groups  |  median size={sorted(sizes)[len(sizes)//2]}  "
          f"max={max(sizes)}  single-frame={sum(1 for s in sizes if s==1)}")
    print(f"  Device: {device}\n")

    embs = extract_embeddings(val_entries, device)

    # Sweep k=1..15
    print(f"  {'k':>4}  {'recall':>8}  {'selection':>10}  {'note'}")
    print("  " + "-" * 40)
    for k in range(1, 16):
        r, s = score_top_k(embs, labels, groups, k)
        note = "<-- top-k" if k == top_k else ""
        print(f"  {k:>4}  {r:>8.1%}  {s:>10.1%}  {note}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3,
                        help="Highlight this k in the sweep output (default: 3)")
    args = parser.parse_args()
    evaluate_cluster_selection(args.top_k)

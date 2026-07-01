"""
One-time preprocessing: extract EfficientNet-B3 embeddings for every photo
and group them into burst sequences (chronological order, 1-second gaps).

Outputs (in embeddings/ directory):
  b3_embeddings.pt        — dict {raw_path_str: embedding_tensor (1536-d float16)}
  burst_sequences.json    — list of bursts, each: {split, photos: [{raw, label}]}

Usage:
    python extract_b3_embeddings.py
"""
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINTS_DIR, MAPPING_FILE, THUMB_SIZE
from data.mapping import (
    _build_ts_raw_map, _group_into_bursts, _ts_str_to_float, flat_entries, load_mapping
)
from data.raw_reader import extract_thumbnail
from models.culling.model import build_model

EMB_DIR  = Path(__file__).parent / "embeddings"
EMB_FILE = EMB_DIR / "b3_embeddings.pt"
SEQ_FILE = EMB_DIR / "burst_sequences.json"

BACKBONE  = "efficientnet_b3"
CKPT_PATH = CHECKPOINTS_DIR / f"culling_{BACKBONE}.pt"
BATCH     = 32


class _FlatDataset(Dataset):
    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = extract_thumbnail(self.paths[idx], size=THUMB_SIZE)
        return self.transform(img), self.paths[idx]


def extract_embeddings(paths: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)

    # Load backbone without the classifier head
    import timm
    model = timm.create_model(BACKBONE, pretrained=False, num_classes=0)
    state = {k: v for k, v in ckpt["model_state"].items()
             if not k.startswith("classifier")}
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds     = _FlatDataset(paths, tf)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=4, pin_memory=True)

    emb_dict: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for imgs, raw_paths in tqdm(loader, desc="Extracting B3 embeddings"):
            embs = model(imgs.to(device)).cpu().to(torch.float16)
            for path, emb in zip(raw_paths, embs):
                emb_dict[path] = emb

    return emb_dict


def build_burst_sequences(mapping: dict, emb_paths: set[str]) -> list[dict]:
    """
    Reconstruct burst sequences by re-reading EXIF timestamps per directory.
    Returns list of {split, photos: [{raw, label}]}.
    """
    # Collect all valid entries grouped by parent directory
    from collections import defaultdict
    dir_entries: dict[str, list] = defaultdict(list)
    for e in flat_entries(mapping):
        if e["raw"] and e["label"] is not None and e["raw"] in emb_paths:
            dir_entries[str(Path(e["raw"]).parent)].append(e)

    print(f"Grouping {sum(len(v) for v in dir_entries.values()):,} photos "
          f"from {len(dir_entries)} directories into burst sequences…")

    sequences: list[dict] = []

    for dir_path_str, entries in tqdm(dir_entries.items(), desc="Building bursts"):
        dir_path = Path(dir_path_str)
        raw_files = [Path(e["raw"]) for e in entries]
        label_map = {e["raw"]: e["label"] for e in entries}
        split_map = {e["raw"]: e["split"] for e in entries}

        # Read EXIF timestamps for this directory
        ts_raw = _build_ts_raw_map(dir_path)
        raw_timestamps: dict[str, float] = {}
        for ts_str, raw_f in ts_raw.items():
            ts_f = _ts_str_to_float(ts_str)
            if ts_f is not None:
                raw_timestamps[str(raw_f)] = ts_f

        # Group into bursts using the same 1-second logic as mapping.py
        burst_groups = _group_into_bursts(raw_files, raw_timestamps)

        for group in burst_groups:
            photos = []
            for f in group:
                raw_str = str(f)
                if raw_str not in label_map:
                    continue
                photos.append({"raw": raw_str, "label": label_map[raw_str]})
            if not photos:
                continue
            # Split for the burst = majority split of its photos
            splits = [split_map[p["raw"]] for p in photos]
            burst_split = max(set(splits), key=splits.count)
            sequences.append({"split": burst_split, "photos": photos})

    return sequences


if __name__ == "__main__":
    EMB_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mapping = load_mapping()
    all_paths = [e["raw"] for e in flat_entries(mapping)
                 if e["raw"] and e["label"] is not None]
    print(f"Total photos to embed: {len(all_paths):,}")

    if EMB_FILE.exists():
        print(f"Embeddings already exist at {EMB_FILE} — skipping extraction.")
        emb_dict = torch.load(str(EMB_FILE), map_location="cpu", weights_only=False)
    else:
        emb_dict = extract_embeddings(all_paths, device)
        torch.save(emb_dict, str(EMB_FILE))
        print(f"Saved {len(emb_dict):,} embeddings -> {EMB_FILE}")

    if SEQ_FILE.exists():
        print(f"Burst sequences already exist at {SEQ_FILE} — skipping.")
    else:
        sequences = build_burst_sequences(mapping, set(emb_dict.keys()))
        with open(SEQ_FILE, "w") as fh:
            json.dump(sequences, fh)
        n_train = sum(1 for s in sequences if s["split"] == "train")
        n_val   = sum(1 for s in sequences if s["split"] == "val")
        n_test  = sum(1 for s in sequences if s["split"] == "test")
        print(f"Saved {len(sequences):,} burst sequences -> {SEQ_FILE}")
        print(f"  train={n_train:,}  val={n_val:,}  test={n_test:,}")

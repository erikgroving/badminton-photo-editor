"""
PyTorch Dataset for burst-sequence LSTM training.

Each sample is one burst: an ordered sequence of (embedding, label) pairs.
The DataLoader collate function pads variable-length sequences.
"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

_EMB_FILE = Path(__file__).parent.parent / "embeddings" / "b3_embeddings.pt"
_SEQ_FILE = Path(__file__).parent.parent / "embeddings" / "burst_sequences.json"


class BurstSequenceDataset(Dataset):
    def __init__(self, split: str, emb_file: Path = _EMB_FILE,
                 seq_file: Path = _SEQ_FILE):
        with open(seq_file) as fh:
            all_seqs = json.load(fh)
        self.seqs = [s for s in all_seqs if s["split"] == split]
        # Load embeddings as float32 (stored as float16 to save disk)
        self.emb = torch.load(str(emb_file), map_location="cpu", weights_only=False)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        burst = self.seqs[idx]
        embs   = torch.stack([self.emb[p["raw"]].float() for p in burst["photos"]])
        labels = torch.tensor([p["label"] for p in burst["photos"]], dtype=torch.float32)
        return embs, labels  # (seq_len, 1536), (seq_len,)


def collate_sequences(batch: list[tuple]) -> tuple:
    """
    Pads a list of variable-length (embs, labels) to the longest sequence.
    Returns (padded_embs, padded_labels, lengths) — sorted longest-first
    for pack_padded_sequence compatibility.
    """
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    lengths = torch.tensor([x[0].shape[0] for x in batch])
    max_len = int(lengths[0])
    emb_dim = batch[0][0].shape[1]

    padded_embs   = torch.zeros(len(batch), max_len, emb_dim)
    padded_labels = torch.zeros(len(batch), max_len)
    for i, (embs, labels) in enumerate(batch):
        L = embs.shape[0]
        padded_embs[i, :L]   = embs
        padded_labels[i, :L] = labels

    return padded_embs, padded_labels, lengths

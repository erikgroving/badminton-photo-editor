"""
BurstLSTM: temporal context on top of frozen EfficientNet-B3 embeddings.

Architecture:
  - Input:  (batch, seq_len, emb_dim=1536) pre-extracted B3 embeddings
  - LSTM:   (1536 -> hidden_size, num_layers) — causal, left-to-right
  - Output: linear(concat(emb[t], h[t])) -> logit at each timestep

The visual embedding is fed directly to the classifier alongside the LSTM
hidden state so the photo-level signal is never lost to the temporal path.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BurstLSTM(nn.Module):
    def __init__(self, emb_dim: int = 1536, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(emb_dim + hidden_size, 1)

    def forward(self, embs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embs:    (batch, max_seq_len, emb_dim) — padded, sorted longest-first
            lengths: (batch,) CPU tensor of actual sequence lengths
        Returns:
            logits:  (batch, max_seq_len) — padded positions are undefined
        """
        packed = pack_padded_sequence(embs, lengths.cpu(), batch_first=True,
                                      enforce_sorted=True)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # lstm_out: (batch, max_seq_len, hidden_size)

        max_len = lstm_out.shape[1]
        concat  = torch.cat([embs[:, :max_len, :], lstm_out], dim=-1)
        return self.classifier(concat).squeeze(-1)  # (batch, max_seq_len)

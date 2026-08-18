"""
model.py

Sequence classifier for ISL word recognition. Takes a sequence of
per-frame landmark vectors (produced by capture.py's extract_landmarks)
and predicts a word class out of a fixed vocabulary.

A bidirectional LSTM is a reasonable, well-understood starting point for
this — it's fast to train on a small vocabulary (~50 words) and doesn't
need much data to start producing sensible results. Swap in a
Transformer later if you have more data and want to push accuracy.
"""

import torch
import torch.nn as nn


class SignClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 num_classes: int = 50, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        lengths: optional (batch,) tensor of true (unpadded) sequence
                 lengths, if you're using variable-length sequences with
                 padding — pass this so the LSTM ignores the pad frames.
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)

        # concat the final forward and backward hidden states
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(final_hidden)

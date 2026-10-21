"""The listener: reads the message, looks at K candidates, and scores each one.

Message -> embeddings -> GRU -> one "meaning" vector m.
Each candidate image -> its own CNN -> vector c_k.
Score_k = m . c_k, and a softmax over the K scores gives the choice.
The dot product means the listener must learn to put the message and the
matching image close together in the same vector space.
"""

from __future__ import annotations

import torch
from torch import nn

from models.vision import VisionEncoder


class Listener(nn.Module):
    def __init__(self, vocab_size: int = 10, feature_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = VisionEncoder(out_dim=feature_dim)
        # Linear on one-hots (instead of nn.Embedding on integers) so gradients reach the speaker.
        self.symbol_embed = nn.Linear(vocab_size, hidden_dim, bias=False)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.to_feature = nn.Linear(hidden_dim, feature_dim)

    def encode_message(self, onehots: torch.Tensor) -> torch.Tensor:
        """onehots: (B, L, V) -> (B, feature_dim)."""
        _, h = self.gru(self.symbol_embed(onehots))
        return self.to_feature(h[-1])

    def forward(self, onehots: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """onehots: (B, L, V); candidates: (B, K, 3, 64, 64) -> logits (B, K)."""
        message = self.encode_message(onehots)
        images = self.encoder(candidates)
        return torch.einsum("bd,bkd->bk", message, images)

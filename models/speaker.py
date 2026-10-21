"""The speaker: looks at the target image and emits L discrete symbols.

The hard part of emergent communication is that symbols are *discrete*: you
cannot backpropagate through "pick symbol 7". We support two standard fixes:

- gumbel (Havrylov & Titov 2017): straight-through Gumbel-softmax. The forward
  pass sends a true one-hot symbol, but the backward pass pretends we sent the
  soft probabilities, so gradients still reach the speaker.
- reinforce: sample symbols and treat the listener's success as a reward.
  No gradient flows through the message; the speaker learns from the policy
  gradient (log-prob of what it said) * (how well it went - baseline).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from models.vision import VisionEncoder


@dataclass
class Message:
    """What the speaker produced for a batch.

    symbols:   (B, L) integer symbols in [0, V).
    onehots:   (B, L, V) one-hot symbols. In gumbel training mode these carry
               straight-through gradients; otherwise they are constants.
    log_probs: (B,) log-probability of the whole message (sum over positions), for REINFORCE.
    entropy:   (B,) mean per-position entropy of the speaker's distribution.
    """

    symbols: torch.Tensor
    onehots: torch.Tensor
    log_probs: torch.Tensor
    entropy: torch.Tensor


def gumbel_softmax_st(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Straight-through Gumbel-softmax, written out so each step is visible.

    1. Add Gumbel noise to the logits: argmax(logits + g) is an exact sample
       from softmax(logits) (the "Gumbel-max trick").
    2. soft = softmax((logits + g) / temperature) is a differentiable
       approximation of that sample; lower temperature = closer to one-hot.
    3. Return hard + (soft - soft.detach()): numerically equal to the hard
       one-hot, but its gradient is the gradient of `soft`.
    """
    uniform = torch.rand_like(logits).clamp_(1e-10, 1.0 - 1e-10)
    gumbel = -torch.log(-torch.log(uniform))
    soft = F.softmax((logits + gumbel) / temperature, dim=-1)
    hard = F.one_hot(soft.argmax(dim=-1), logits.shape[-1]).to(soft.dtype)
    return hard + soft - soft.detach()


class Speaker(nn.Module):
    """CNN features -> initial GRU state -> one symbol per step, fed back as the next input."""

    def __init__(
        self,
        vocab_size: int = 10,
        msg_len: int = 3,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        mode: str = "gumbel",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in {"gumbel", "reinforce"}:
            raise ValueError(f"mode must be 'gumbel' or 'reinforce', got {mode!r}")
        self.vocab_size, self.msg_len = vocab_size, msg_len
        self.mode, self.temperature = mode, temperature
        self.encoder = VisionEncoder(out_dim=feature_dim)
        self.init_hidden = nn.Linear(feature_dim, hidden_dim)
        # A linear map from one-hot to embedding, so straight-through gradients can pass through it.
        self.symbol_embed = nn.Linear(vocab_size, hidden_dim, bias=False)
        self.start = nn.Parameter(torch.zeros(hidden_dim))  # learned "begin message" input
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images: torch.Tensor) -> Message:
        """images: (B, 3, 64, 64). In eval mode the speaker is greedy (argmax) and deterministic."""
        h = torch.tanh(self.init_hidden(self.encoder(images)))
        x = self.start.expand(images.shape[0], -1)
        symbols, onehots, log_probs, entropies = [], [], [], []
        for _ in range(self.msg_len):
            h = self.gru(x, h)
            logits = self.out(h)
            dist = Categorical(logits=logits)
            if not self.training:
                sym = logits.argmax(dim=-1)
                onehot = F.one_hot(sym, self.vocab_size).float()
            elif self.mode == "gumbel":
                onehot = gumbel_softmax_st(logits, self.temperature)
                sym = onehot.argmax(dim=-1)
            else:
                sym = dist.sample()
                onehot = F.one_hot(sym, self.vocab_size).float()
            symbols.append(sym)
            onehots.append(onehot)
            log_probs.append(dist.log_prob(sym))
            entropies.append(dist.entropy())
            x = self.symbol_embed(onehot)
        return Message(
            symbols=torch.stack(symbols, dim=1),
            onehots=torch.stack(onehots, dim=1),
            log_probs=torch.stack(log_probs, dim=1).sum(dim=1),
            entropy=torch.stack(entropies, dim=1).mean(dim=1),
        )

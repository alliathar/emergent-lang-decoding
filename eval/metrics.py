"""Evaluation: does the language work, is it actually used, and is it structured?

- accuracy:  how often the listener picks the target (compare to chance = 1/K).
- ablation:  replace the speaker's message with random symbols or a fixed message.
             If accuracy falls to ~chance, the listener really relies on the message.
- topsim:    topographic similarity. Do similar images get similar messages?
- symbol analysis: which symbol in which position tracks which attribute?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from data.shapes import ATTRIBUTE_NAMES, ATTRIBUTES, ShapeSplit
from models.listener import Listener
from models.speaker import Speaker

MESSAGE_MODES = ("speaker", "random", "fixed")


# ---------------------------------------------------------------------------
# Playing games
# ---------------------------------------------------------------------------

def make_messages(
    mode: str,
    speaker: Speaker,
    target_images: torch.Tensor,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """One-hot messages (B, L, V) for a batch of targets.

    mode="speaker": what the speaker actually says.
    mode="random":  uniformly random symbols (ignores the image entirely).
    mode="fixed":   the same message (all symbol 0) for every image.
    """
    batch, length, vocab = target_images.shape[0], speaker.msg_len, speaker.vocab_size
    if mode == "speaker":
        return speaker(target_images).onehots
    if mode == "random":
        symbols = torch.randint(0, vocab, (batch, length), generator=rng, device="cpu").to(target_images.device)
    elif mode == "fixed":
        symbols = torch.zeros(batch, length, dtype=torch.long, device=target_images.device)
    else:
        raise ValueError(f"unknown message mode {mode!r}")
    return F.one_hot(symbols, vocab).float()


@torch.no_grad()
def play(
    speaker: Speaker,
    listener: Listener,
    split: ShapeSplit,
    candidates: np.ndarray,
    target_pos: np.ndarray,
    message_mode: str = "speaker",
    batch_size: int = 250,
    seed: int = 0,
) -> dict[str, float]:
    """Play a fixed set of games and return {"acc": ..., "loss": ...}."""
    speaker.eval()
    listener.eval()
    rng = torch.Generator().manual_seed(seed)
    device = split.images.device
    correct, loss_sum = 0, 0.0
    for start in range(0, len(candidates), batch_size):
        cand = candidates[start : start + batch_size]
        pos = torch.as_tensor(target_pos[start : start + batch_size], device=device)
        targets = split.float_images(cand[np.arange(len(cand)), pos.cpu().numpy()])
        onehots = make_messages(message_mode, speaker, targets, rng)
        logits = listener(onehots, split.float_images(cand))
        correct += (logits.argmax(dim=1) == pos).sum().item()
        loss_sum += F.cross_entropy(logits, pos, reduction="sum").item()
    return {"acc": correct / len(candidates), "loss": loss_sum / len(candidates)}


@torch.no_grad()
def speaker_messages(speaker: Speaker, split: ShapeSplit, batch_size: int = 500) -> np.ndarray:
    """The speaker's (greedy) message for every image in the split: (N, L) ints."""
    speaker.eval()
    out = [
        speaker(split.float_images(np.arange(start, min(start + batch_size, len(split))))).symbols.cpu()
        for start in range(0, len(split), batch_size)
    ]
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------
# Topographic similarity
# ---------------------------------------------------------------------------

def edit_distance(a: tuple[int, ...] | np.ndarray, b: tuple[int, ...] | np.ndarray) -> int:
    """Levenshtein distance: fewest insertions, deletions or substitutions turning a into b."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def message_distances(messages: np.ndarray) -> np.ndarray:
    """Condensed (pdist-style) edit distances between all pairs of messages.

    Many images share a message, so we compute edit distance once per pair of
    *unique* messages and then look it up, which is far faster than doing every pair.
    """
    unique, inverse = np.unique(messages, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    n_u = len(unique)
    table = np.zeros((n_u, n_u))
    for i in range(n_u):
        for j in range(i + 1, n_u):
            table[i, j] = table[j, i] = edit_distance(unique[i], unique[j])
    rows, cols = np.triu_indices(len(messages), k=1)
    return table[inverse[rows], inverse[cols]]


def topographic_similarity(
    attrs: np.ndarray, messages: np.ndarray, max_items: int = 1000, seed: int = 0
) -> float:
    """Spearman correlation between attribute distance (Hamming) and message distance (edit).

    1.0 means "images that differ in more attributes always get more different
    messages" — a perfectly structured, compositional-looking language.
    0 means message similarity tells you nothing about image similarity.
    Returns 0.0 when either distance is constant (correlation is undefined).
    """
    if len(attrs) > max_items:
        idx = np.random.default_rng(seed).choice(len(attrs), size=max_items, replace=False)
        attrs, messages = attrs[idx], messages[idx]
    attr_dist = pdist(attrs, metric="hamming")
    msg_dist = message_distances(messages)
    if np.ptp(attr_dist) == 0 or np.ptp(msg_dist) == 0:
        return 0.0
    return float(spearmanr(attr_dist, msg_dist).statistic)


# ---------------------------------------------------------------------------
# Symbol <-> attribute analysis
# ---------------------------------------------------------------------------

def entropy(x: np.ndarray) -> float:
    """Shannon entropy (bits) of a discrete variable given as an array of samples."""
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """I(X; Y) = H(X) + H(Y) - H(X, Y), in bits."""
    joint = np.stack([x, y], axis=1)
    _, joint_ids = np.unique(joint, axis=0, return_inverse=True)
    return entropy(x) + entropy(y) - entropy(joint_ids.reshape(-1))


def symbol_attribute_mi(messages: np.ndarray, attrs: np.ndarray) -> pd.DataFrame:
    """Normalised MI between each message position and each attribute.

    Value = I(symbol at position; attribute) / H(attribute), so 1.0 means the
    symbol at that position fully determines the attribute and 0 means it says
    nothing about it. Rows: positions, columns: attributes.
    """
    data = {
        name: [mutual_information(messages[:, p], attrs[:, a]) / max(entropy(attrs[:, a]), 1e-12)
               for p in range(messages.shape[1])]
        for a, name in enumerate(ATTRIBUTE_NAMES)
    }
    return pd.DataFrame(data, index=pd.Index([f"pos{p}" for p in range(messages.shape[1])], name="message_position"))


def cooccurrence(messages: np.ndarray, attrs: np.ndarray, vocab_size: int) -> list[pd.DataFrame]:
    """For each position: a (V symbols x all attribute values) table of P(value | symbol).

    Read a row as "when the speaker says this symbol here, what is the image like?".
    Rows for symbols that are never used are NaN.
    """
    columns = [f"{name}={value}" for name, values in ATTRIBUTES.items() for value in values]
    tables = []
    for p in range(messages.shape[1]):
        table = np.full((vocab_size, len(columns)), np.nan)
        for s in range(vocab_size):
            mask = messages[:, p] == s
            if mask.any():
                table[s] = np.concatenate(
                    [np.bincount(attrs[mask, a], minlength=len(values)) / mask.sum()
                     for a, values in enumerate(ATTRIBUTES.values())]
                )
        tables.append(pd.DataFrame(table, columns=columns, index=pd.Index(range(vocab_size), name="symbol")))
    return tables


def symbol_table(messages: np.ndarray, attrs: np.ndarray, vocab_size: int) -> pd.DataFrame:
    """One row per (position, symbol) that is used: the attribute value it best predicts.

    "Best" = largest gain of P(value | symbol) over the base rate P(value), so a
    2-valued attribute like size doesn't win just because its base rate is 50%.
    """
    rows = []
    for p, table in enumerate(cooccurrence(messages, attrs, vocab_size)):
        base = np.concatenate(
            [np.bincount(attrs[:, a], minlength=len(v)) / len(attrs) for a, v in enumerate(ATTRIBUTES.values())]
        )
        for s, probs in table.iterrows():
            if probs.isna().all():
                continue
            gain = probs.to_numpy() - base
            best = int(np.argmax(gain))
            attribute, value = table.columns[best].split("=")
            rows.append({
                "position": p,
                "symbol": s,
                "count": int((messages[:, p] == s).sum()),
                "best_attribute": attribute,
                "best_value": value,
                "p_value_given_symbol": round(float(probs.iloc[best]), 4),
                "p_value_base_rate": round(float(base[best]), 4),
            })
    return pd.DataFrame(rows)

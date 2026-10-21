"""Procedurally generated shapes for the referential game.

Because we draw every image ourselves, we know the true attributes of each one.
That is what makes the later analysis possible: we can ask "does symbol 7 in
position 0 mean *red*?" only if we know which images are red.

Each image is 64x64 RGB on a black background and has four attributes:
shape, colour, size and position (with a small random jitter so that two images
of the same combination are not pixel-identical).

Splits
------
- train / val / test: drawn from the *training* attribute combinations.
  val is for monitoring during training, test is the final report number.
- heldout: drawn only from attribute combinations never seen in training.
  We hold out whole (shape, colour) pairs, e.g. "blue triangle", so the agents
  have seen blue things and triangles but never a blue triangle. If the language
  is compositional, the speaker should still describe it sensibly.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import torch

IMAGE_SIZE = 64

# Order matters: an image's attributes are stored as an integer vector in this order.
ATTRIBUTES: dict[str, list[str]] = {
    "shape": ["circle", "square", "triangle", "cross"],
    "colour": ["red", "green", "blue", "yellow", "magenta"],
    "size": ["small", "large"],
    "position": ["left", "centre", "right"],
}
ATTRIBUTE_NAMES: list[str] = list(ATTRIBUTES)

COLOUR_RGB: dict[str, tuple[int, int, int]] = {
    "red": (230, 40, 40),
    "green": (40, 200, 60),
    "blue": (50, 90, 240),
    "yellow": (240, 220, 40),
    "magenta": (220, 50, 220),
}
SIZE_RADIUS: dict[str, int] = {"small": 7, "large": 12}
POSITION_X: dict[str, int] = {"left": 16, "centre": 32, "right": 48}

# A pixel grid reused by every drawing call (row = y, column = x).
_YY, _XX = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _shape_mask(shape: str, cx: float, cy: float, r: float) -> np.ndarray:
    """Boolean mask of the pixels covered by `shape` centred at (cx, cy) with 'radius' r."""
    dx, dy = _XX - cx, _YY - cy
    if shape == "circle":
        return dx**2 + dy**2 <= r**2
    if shape == "square":
        # Slightly smaller than the bounding box so a square and a circle cover similar areas.
        half = 0.85 * r
        return (np.abs(dx) <= half) & (np.abs(dy) <= half)
    if shape == "triangle":
        # Upward-pointing: zero width at the top (dy = -r), full width r at the bottom (dy = +r).
        half_width = (dy + r) / 2
        return (dy >= -r) & (dy <= r) & (np.abs(dx) <= half_width)
    if shape == "cross":
        arm = r / 3
        horizontal = (np.abs(dy) <= arm) & (np.abs(dx) <= r)
        vertical = (np.abs(dx) <= arm) & (np.abs(dy) <= r)
        return horizontal | vertical
    raise ValueError(f"unknown shape {shape!r}")


def render(attrs: dict[str, str], rng: np.random.Generator, jitter: int = 2) -> np.ndarray:
    """Draw one image as a (64, 64, 3) uint8 array.

    The jitter shifts the centre by up to `jitter` pixels in x and y, so the
    agents cannot memorise exact pixels and must recognise the attributes.
    The shift is continuous (sub-pixel), which changes the shape's edge pixels,
    so test images are new instances rather than copies of training images.
    """
    r = SIZE_RADIUS[attrs["size"]]
    cx = POSITION_X[attrs["position"]] + rng.uniform(-jitter, jitter)
    cy = IMAGE_SIZE // 2 + rng.uniform(-jitter, jitter)
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    image[_shape_mask(attrs["shape"], cx, cy, r)] = COLOUR_RGB[attrs["colour"]]
    return image


def attrs_to_dict(attr_ids: tuple[int, ...] | np.ndarray) -> dict[str, str]:
    """(0, 2, 1, 0) -> {"shape": "circle", "colour": "blue", "size": "large", "position": "left"}."""
    return {name: ATTRIBUTES[name][int(i)] for name, i in zip(ATTRIBUTE_NAMES, attr_ids)}


def make_sample(
    attr_ids: tuple[int, ...], rng: np.random.Generator, jitter: int = 2
) -> tuple[torch.Tensor, dict[str, str]]:
    """Generate one sample: a float (3, 64, 64) tensor in [0, 1] and its attribute dict."""
    attrs = attrs_to_dict(attr_ids)
    image = torch.from_numpy(render(attrs, rng, jitter)).permute(2, 0, 1).float() / 255.0
    return image, attrs


# ---------------------------------------------------------------------------
# Combinations and the held-out split
# ---------------------------------------------------------------------------

def all_combinations() -> list[tuple[int, ...]]:
    """Every attribute combination as a tuple of value indices (4*5*2*3 = 120)."""
    return list(itertools.product(*(range(len(v)) for v in ATTRIBUTES.values())))


def choose_heldout_pairs(holdout_frac: float, seed: int) -> list[tuple[int, int]]:
    """Pick (shape, colour) pairs that never appear in training.

    We insist that every shape and every colour still appears in training with
    *some* partner; otherwise the held-out test would ask about a word the agents
    have never had a chance to invent.
    """
    n_shapes, n_colours = len(ATTRIBUTES["shape"]), len(ATTRIBUTES["colour"])
    pairs = [(s, c) for s in range(n_shapes) for c in range(n_colours)]
    n_hold = round(holdout_frac * len(pairs))
    if n_hold == 0:
        return []
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        chosen = [pairs[i] for i in rng.choice(len(pairs), size=n_hold, replace=False)]
        kept = set(pairs) - set(chosen)
        if {s for s, _ in kept} == set(range(n_shapes)) and {c for _, c in kept} == set(range(n_colours)):
            return sorted(chosen)
    raise ValueError(f"holdout_frac={holdout_frac} is too large to keep every shape and colour in training")


def split_combinations(holdout_frac: float, seed: int) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Return (train_combos, heldout_combos), splitting on (shape, colour) pairs."""
    heldout_pairs = set(choose_heldout_pairs(holdout_frac, seed))
    train, heldout = [], []
    for combo in all_combinations():
        (heldout if (combo[0], combo[1]) in heldout_pairs else train).append(combo)
    return train, heldout


# ---------------------------------------------------------------------------
# A split of images + game sampling
# ---------------------------------------------------------------------------

@dataclass
class ShapeSplit:
    """A fixed pool of images (uint8, N x 3 x 64 x 64) and their attribute indices (N x 4)."""

    name: str
    images: torch.Tensor
    attrs: np.ndarray
    combo_ids: np.ndarray = field(init=False)
    _by_combo: dict[int, np.ndarray] = field(init=False, repr=False)
    _hard_pool: dict[int, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Encode each attribute vector as one integer so "same combination?" is a cheap comparison.
        sizes = [len(v) for v in ATTRIBUTES.values()]
        self.combo_ids = np.ravel_multi_index(self.attrs.T, sizes)
        self._by_combo = {int(c): np.flatnonzero(self.combo_ids == c) for c in np.unique(self.combo_ids)}
        self._hard_pool = {}

    def __len__(self) -> int:
        return len(self.attrs)

    def to(self, device: torch.device | str) -> "ShapeSplit":
        """Move the image tensor to `device` (the whole dataset is small enough to live there)."""
        self.images = self.images.to(device)
        return self

    def float_images(self, idx: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Index images and convert to float in [0, 1]. Keeps any leading index shape."""
        idx_t = torch.as_tensor(idx, device=self.images.device)
        flat = self.images[idx_t.reshape(-1)].float() / 255.0
        return flat.reshape(*idx_t.shape, *self.images.shape[1:])

    def hard_pool(self, combo_id: int) -> np.ndarray:
        """Indices of images whose combination shares >= 2 attributes with `combo_id` (but isn't it)."""
        if combo_id not in self._hard_pool:
            sizes = [len(v) for v in ATTRIBUTES.values()]
            target = np.array(np.unravel_index(combo_id, sizes))
            shared = (self.attrs == target).sum(axis=1)
            self._hard_pool[combo_id] = np.flatnonzero((shared >= 2) & (self.combo_ids != combo_id))
        return self._hard_pool[combo_id]

    def sample_games(
        self, batch_size: int, n_candidates: int, hard: bool, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample `batch_size` games.

        Returns (candidates, target_pos): candidates is (B, K) image indices and
        target_pos is (B,) the column holding the target. The target sits at a
        random column, so its position carries no information.

        Distractors never have exactly the target's attribute combination:
        such a game would only be solvable by reading jitter, which is not what
        we want the language to describe.
        """
        targets = rng.integers(0, len(self), size=batch_size)
        if hard:
            distractors = np.stack([self._hard_distractors(int(self.combo_ids[t]), n_candidates - 1, rng) for t in targets])
        else:
            distractors = self._random_distractors(targets, n_candidates - 1, rng)
        candidates = np.concatenate([targets[:, None], distractors], axis=1)
        # Shuffle each row so the target isn't always in column 0.
        order = rng.permuted(np.tile(np.arange(n_candidates), (batch_size, 1)), axis=1)
        candidates = np.take_along_axis(candidates, order, axis=1)
        target_pos = np.argmax(order == 0, axis=1)
        return candidates, target_pos

    def _random_distractors(self, targets: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        if len(self._by_combo) < 2:
            raise ValueError(f"split {self.name!r} has a single combination; no valid distractors")
        out = rng.integers(0, len(self), size=(len(targets), n))
        # Rejection sampling: redraw any distractor that matches its target's combination.
        bad = self.combo_ids[out] == self.combo_ids[targets][:, None]
        while bad.any():
            out[bad] = rng.integers(0, len(self), size=int(bad.sum()))
            bad = self.combo_ids[out] == self.combo_ids[targets][:, None]
        return out

    def _hard_distractors(self, combo_id: int, n: int, rng: np.random.Generator) -> np.ndarray:
        pool = self.hard_pool(combo_id)
        if len(pool) == 0:
            raise ValueError(f"no hard distractors for combination {combo_id} in split {self.name!r}")
        return rng.choice(pool, size=n, replace=len(pool) < n)


def generate_split(
    name: str, combos: list[tuple[int, ...]], n_images: int, seed: int, jitter: int = 2
) -> ShapeSplit:
    """Generate `n_images` images, choosing combinations uniformly from `combos`.

    Deterministic: the same (combos, n_images, seed, jitter) always gives the same pixels.
    """
    rng = np.random.default_rng(seed)
    chosen = np.asarray(combos)[rng.integers(0, len(combos), size=n_images)]
    images = np.stack([render(attrs_to_dict(c), rng, jitter) for c in chosen])
    images_t = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    return ShapeSplit(name=name, images=images_t, attrs=chosen.astype(np.int64))


def build_splits(
    n_train: int = 3000,
    n_val: int = 500,
    n_test: int = 1000,
    n_heldout: int = 1000,
    holdout_frac: float = 0.2,
    jitter: int = 2,
    seed: int = 0,
) -> dict[str, ShapeSplit]:
    """Build the four splits. Each split gets its own derived seed so they don't share images."""
    train_combos, heldout_combos = split_combinations(holdout_frac, seed)
    sizes = {"train": n_train, "val": n_val, "test": n_test}
    splits = {
        name: generate_split(name, train_combos, n, seed=seed * 1000 + i, jitter=jitter)
        for i, (name, n) in enumerate(sizes.items())
    }
    if heldout_combos:
        splits["heldout"] = generate_split("heldout", heldout_combos, n_heldout, seed=seed * 1000 + 3, jitter=jitter)
    return splits

import numpy as np
import torch

from data.shapes import ATTRIBUTES, all_combinations, build_splits, make_sample, split_combinations

SMALL = dict(n_train=300, n_val=50, n_test=100, n_heldout=100)


def test_make_sample_returns_tensor_and_attrs():
    image, attrs = make_sample((0, 1, 0, 2), np.random.default_rng(0))
    assert image.shape == (3, 64, 64) and image.dtype == torch.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert attrs == {"shape": "circle", "colour": "green", "size": "small", "position": "right"}
    assert image.sum() > 0  # something was drawn


def test_dataset_is_deterministic():
    a, b = build_splits(seed=3, **SMALL), build_splits(seed=3, **SMALL)
    for name in a:
        assert torch.equal(a[name].images, b[name].images)
        assert np.array_equal(a[name].attrs, b[name].attrs)
    c = build_splits(seed=4, **SMALL)
    assert not torch.equal(a["train"].images, c["train"].images)


def test_heldout_combinations_never_in_training():
    train, heldout = split_combinations(0.2, seed=0)
    assert len(train) + len(heldout) == len(all_combinations())
    assert set(train).isdisjoint(heldout)
    assert len({(s, c) for s, c, *_ in heldout}) == 4  # 20% of the 20 (shape, colour) pairs
    # Every shape and colour value still occurs in training.
    assert {c[0] for c in train} == set(range(len(ATTRIBUTES["shape"])))
    assert {c[1] for c in train} == set(range(len(ATTRIBUTES["colour"])))

    splits = build_splits(seed=0, **SMALL)
    pairs = lambda s: {tuple(a[:2]) for a in s.attrs}  # noqa: E731
    for name in ("train", "val", "test"):
        assert pairs(splits[name]).isdisjoint(pairs(splits["heldout"]))


def test_test_images_are_new_instances():
    # val/test use the training combinations, but sub-pixel jitter should make
    # (almost) every image a new instance rather than a copy of a training image.
    splits = build_splits(seed=0, **SMALL)
    train_pixels = {img.numpy().tobytes() for img in splits["train"].images}
    for name in ("val", "test"):
        copies = sum(img.numpy().tobytes() in train_pixels for img in splits[name].images)
        assert copies / len(splits[name]) < 0.05, name


def test_distractors():
    split = build_splits(seed=0, **SMALL)["train"]
    rng = np.random.default_rng(0)
    for hard in (False, True):
        cand, pos = split.sample_games(64, 4, hard, rng)
        assert cand.shape == (64, 4) and pos.shape == (64,)
        target_attrs = split.attrs[cand[np.arange(64), pos]]
        for k in range(4):
            mask = pos != k
            shared = (split.attrs[cand[mask, k]] == target_attrs[mask]).sum(axis=1)
            assert (shared < 4).all()  # never the target's exact combination
            if hard:
                assert (shared >= 2).all()
    # The target lands in every column at some point.
    _, pos = split.sample_games(500, 4, False, rng)
    assert set(pos) == {0, 1, 2, 3}

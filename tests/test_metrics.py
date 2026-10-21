import itertools

import numpy as np

from data.shapes import ATTRIBUTES
from eval.metrics import edit_distance, mutual_information, symbol_attribute_mi, topographic_similarity

ALL_ATTRS = np.array(list(itertools.product(*(range(len(v)) for v in ATTRIBUTES.values()))))


def compositional_language(attrs: np.ndarray) -> np.ndarray:
    """One position per attribute; each position uses its own block of symbols (shape 0-3, colour 4-8, ...)."""
    offsets = np.cumsum([0] + [len(v) for v in ATTRIBUTES.values()])[:-1]
    return attrs + offsets


def test_edit_distance():
    assert edit_distance((1, 2, 3), (1, 2, 3)) == 0
    assert edit_distance((1, 2, 3), (1, 9, 3)) == 1
    assert edit_distance((1, 2, 3), (2, 3, 1)) == 2  # delete the leading 1, append it at the end
    assert edit_distance((), (1, 2)) == 2


def test_topsim_perfectly_compositional_is_one():
    assert topographic_similarity(ALL_ATTRS, compositional_language(ALL_ATTRS)) > 0.99


def test_topsim_random_language_is_about_zero():
    rng = np.random.default_rng(0)
    random_messages = rng.integers(0, 10, size=(len(ALL_ATTRS), 4))
    assert abs(topographic_similarity(ALL_ATTRS, random_messages)) < 0.1


def test_topsim_constant_language_is_zero():
    assert topographic_similarity(ALL_ATTRS, np.zeros((len(ALL_ATTRS), 3), dtype=int)) == 0.0


def test_mutual_information():
    x = np.array([0, 0, 1, 1] * 25)
    assert np.isclose(mutual_information(x, x), 1.0)  # 1 bit
    assert np.isclose(mutual_information(x, np.array([0, 1] * 50)), 0.0)


def test_symbol_attribute_mi_finds_the_right_position():
    mi = symbol_attribute_mi(compositional_language(ALL_ATTRS), ALL_ATTRS)
    # Position i fully determines attribute i and says nothing about the others.
    assert np.allclose(mi.to_numpy(), np.eye(4), atol=1e-9)

"""Save a grid of generated images, to check by eye that the dataset looks right.

Usage:
    python scripts/show_samples.py                     # results/samples_train.png
    python scripts/show_samples.py --split heldout --n 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.shapes import attrs_to_dict, build_splits  # noqa: E402
from eval.plots import plot_image_grid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "heldout"])
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--data_seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    split = build_splits(seed=args.data_seed)[args.split]
    n = min(args.n, len(split))
    titles = ["\n".join(attrs_to_dict(a).values()) for a in split.attrs[:n]]
    out = args.out or PROJECT_ROOT / "results" / f"samples_{args.split}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_image_grid(split.images[:n].numpy(), titles, out)
    print(f"saved {n} {args.split} samples to {out}")


if __name__ == "__main__":
    main()

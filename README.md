# Emergent Language Referential Game — Level 1 (synthetic shapes)

Two neural agents, a **speaker** and a **listener**, each with their own CNN trained from
scratch, play a referential game. The speaker sees a target image and sends a message of
`L` symbols from a vocabulary of `V`. The listener reads the message, looks at `K` candidate
images and must pick the target. We study the "language" they invent. The full spec is in
[CLAUDE.md](CLAUDE.md).

## Setup

```
git clone https://github.com/alliathar/emergent-lang-decoding.git
cd emergent-lang-decoding
pip install -r requirements.txt
python -m pytest        # ~1 minute, includes a short training run
```

For a CPU-only machine, `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
is a much smaller download. A GPU is used automatically when one is available (`--device cpu` forces the CPU).

## Common commands

| What | Command |
|---|---|
| Look at the dataset | `python scripts/show_samples.py` (add `--split heldout`) |
| Default run (V=10, L=3, K=4, gumbel) | `python train.py` |
| Change settings | `python train.py --seed 1 --vocab_size 20 --mode reinforce --hard_distractors true` |
| Use another config | `python train.py --config configs/reinforce.yaml` |
| No-communication baseline | `python train.py --config configs/no_comm.yaml` |
| Sweep (configs × seeds) | `python scripts/sweep.py configs/sweep_vocab_len.yaml` |
| Tests | `python -m pytest` |

Every key in `DEFAULT_CONFIG` ([train.py](train.py)) can be set in a YAML config or as a `--flag`.
`seed` controls training. `data_seed` controls the images and held-out pairs, so every training
seed is evaluated on the same data.

## What a run writes (`results/<run_name>/`)

- `config.yaml`: the exact settings used
- `metrics.csv` and `training_curves.png`: train/val accuracy, loss and speaker entropy per epoch
- `checkpoint.pt`: speaker and listener weights
- `final.json`: the headline numbers for `test` and `heldout`:
  - `acc` vs `chance` (= 1/K)
  - `acc_random_msg`, `acc_fixed_msg`: message ablations; ~chance means the message is really used
  - `topsim`: topographic similarity (Spearman of attribute Hamming vs message edit distance)
  - `n_unique_messages`
- `test/` and `heldout/`:
  - `symbol_attribute_mi.csv/.png`: normalised MI between each message position and each attribute
  - `cooccurrence_pos*.csv/.png`: P(attribute value | symbol) for each position
  - `symbol_table.csv`: the attribute value each (position, symbol) best predicts
  - `examples.csv/.png`: 50 (image, message) pairs

Sweeps write `results/summaries/<name>.csv` (one row per run) and `<name>_agg.csv`
(mean ± std over seeds). Re-running a sweep skips finished runs.

## Splits

- **train / val / test** use the same attribute combinations, with new image instances
  (each image has a random sub-pixel position jitter).
- **heldout** contains only the 4 (shape, colour) pairs (20%) that never appear in training,
  e.g. "red square". It tests whether the language generalises to new combinations.

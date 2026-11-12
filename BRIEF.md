# BRIEF.md — Emergent Language Referential Game (Level 1)

## Project context
We study whether two neural agents invent a shared "language" to identify images, and how that language changes as images get
more realistic. This repository uses simple synthetic shapes, agents learn to see from scratch.

Key papers:
- Lazaridou et al. 2017 — referential game, single symbol (arXiv 1612.07182)
- Havrylov & Titov 2017 — sequences of symbols, Gumbel-softmax (arXiv 1705.11192)
- Lazaridou et al. 2018 — raw pixel input, structure → compositionality (arXiv 1804.03984)

## Hard constraints
- **From scratch in PyTorch.** No pretrained models in Level 1. No high-level emergent-comm libraries.
- **Must run on a laptop CPU** for small configs (a run should finish in < 15 min on CPU).
  GPU (Kaggle T4) is optional for full sweeps; code must auto-detect `cuda` and fall back to `cpu`.
- **Windows-friendly.** Used `pathlib`, no shell-specific scripts, `num_workers=0` by default.
- Zero cost: no paid APIs or services.
- Reproducible: every run takes a `--seed`; seed Python, NumPy and torch.

## A single round of the game looks like this:
1. Sample K candidate images (default K=4) where one is the target.
2. **Speaker** sees only the target → emits a message: sequence of length L (default 3)
   of integer symbols from vocab size V (default 10). Symbols start meaningless.
3. **Listener** sees the message + all K candidates (shuffled) → picks one.
4. Both succeed if the pick is the target. Shared loss = cross-entropy on listener's choice.

## Dataset: generated shapes (`data/shapes.py`)
Generate images procedurally so we know the true attributes of every image.
- The images are 64×64 RGB with a plain black background.
- Attributes: shape {circle, square, triangle, cross}, colour {red, green, blue, yellow, magenta},
  size {small, large}, position {left, centre, right} (small jitter allowed).
- Return `(image_tensor, attribute_dict)`; generation must be deterministic given a seed.
- **Held-out split:** hold out ~20% of attribute *combinations* (e.g. never train on
  "blue triangle"), to test whether the language generalises to new combinations i.e. to be able to test compositionality.
- Distractors: random by default; add an option for "hard" distractors that share 2+ attributes
  with the target.

## Models (`models/`)
- `vision.py` — small CNN encoder (3–4 conv layers → vector of size 128). Speaker and listener
  each have **their own** encoder (no weight sharing).
- `speaker.py` — CNN vector → GRU that emits L symbols. Support two training modes:
  - `gumbel` (default): straight-through Gumbel-softmax (Havrylov & Titov). Discrete forward,
    differentiable backward. Temperature is a config value.
  - `reinforce`: sample symbols, REINFORCE with a running-mean baseline + entropy bonus.
- `listener.py` — embeds symbols → GRU → message vector; encodes each candidate with its CNN;
  scores = dot product; softmax over K.

## Evaluation (`eval/`)
Always report on the **test** split and the **held-out combinations** split:
1. **Accuracy** vs chance (1/K).
2. **Message ablation:** replace messages with random symbols / a fixed message → accuracy
   should drop to ~chance. This proves the message is actually used.
3. **Topographic similarity:** Spearman correlation between pairwise attribute distance (Hamming)
   and pairwise message distance (edit distance). Higher = more structured language.
4. **Symbol–attribute analysis:** for each position/symbol, which attribute value it best predicts
   (mutual information or simple co-occurrence table). Save as CSV + heatmap.
5. **Baselines:** random listener (chance) and "no-communication" listener trained with a
   constant message.

## Experiments to support (via config, not code changes)
Vary: V, L, K, distractor difficulty, training mode (gumbel vs reinforce), seeds (≥3 per config).

## Repo layout
```
data/shapes.py         dataset generation + splits
models/vision.py       CNN encoder
models/speaker.py
models/listener.py
train.py               training loop, CLI args, checkpoints, CSV logging
eval/metrics.py        accuracy, ablation, topsim, symbol analysis
eval/plots.py          training curves, heatmaps
configs/*.yaml         experiment configs (default.yaml = small CPU run)
scripts/sweep.py       run many configs/seeds, collect a results table
tests/                 pytest
results/               outputs (git-ignored except summary tables)
```

## Logging and outputs
- Each run writes to `results/<run_name>/`: `config.yaml`, `metrics.csv` (per-epoch),
  `final.json`, checkpoint, and plots.
- Also save ~50 example (target image, message) pairs for the demo/report.

## Tests (keep them fast)
- Dataset is deterministic for a fixed seed and splits don't overlap.
- Speaker output shape is (batch, L) with values in [0, V).
- Topsim returns ~1.0 on a hand-built perfectly compositional toy language and ~0 on a random one.
- A 1-minute smoke run trains without errors and beats chance.

## Build order
1. Dataset + a script to visualise a grid of samples.
2. Models + training with `gumbel`; get accuracy well above chance on CPU.
3. Ablation + topsim + symbol analysis.
4. `reinforce` mode.
5. Sweep script + summary table.


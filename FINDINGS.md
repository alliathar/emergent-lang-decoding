# Level 1 findings so far

Status as of 21 Oct 2026. All results are from **one seed (seed 0)** on a laptop CPU and are
preliminary until the sweeps (3 seeds per config) are run. Setup and commands are in
[README.md](README.md); the spec is in [CLAUDE.md](CLAUDE.md).

## TL;DR

- All five build-order steps are implemented, and the tests pass (19 tests, about 1 minute).
- **Gumbel works:** the default game (V=10, L=3, K=4, random distractors) reaches **94.5% on test**
  and **96.2% on held-out combinations**, against 25% chance. It trains in about 7 minutes on CPU.
- **The message is really used:** replacing it with random symbols or a fixed message drops
  accuracy to about 24%, i.e. chance.
- **The language is not compositional:** the first symbol encodes position (and partly size).
  Colour and shape are spread across *combinations* of symbols, not given their own slot.
  Topsim is moderate: 0.39 on test, 0.34 on held-out.
- **REINFORCE is much worse:** at best 80% after tuning, and it never invents words for colour
  or shape.

## 1. Gumbel default run

`python train.py` (config: [configs/default.yaml](configs/default.yaml)). Outputs are in
`results/gumbel_V10_L3_K4_random_s0/`.

| | Test | Held-out |
|---|---|---|
| Accuracy (chance 0.25) | **0.945** | **0.962** |
| Accuracy, random message | 0.242 | 0.270 |
| Accuracy, fixed message | 0.236 | 0.243 |
| Topographic similarity | 0.39 | 0.34 |
| Distinct messages used | 165 | 135 |

Held-out accuracy is as high as test accuracy. The agents describe combinations they never
trained on (red/magenta squares, green/magenta triangles) as well as ones they did.
The language generalises, even though it isn't neatly compositional (section 3).

## 2. Getting Gumbel to work: the "position-only" trap

With the textbook settings (temperature 1.0, batch 32), the first run plateaued at **61%**
with only **3 distinct messages**. I tried 12 variants. The accuracy plateaus line up exactly
with which attributes the speaker describes:

| Speaker describes | Messages | Expected accuracy (K=4) | Observed plateau |
|---|---|---|---|
| position only | 3 | ~0.6 | 0.61 |
| position + size | 6 | ~0.8 | 0.77–0.81 |
| everything | many | ~1.0 | 0.95 |

Why ~0.6: with 3 messages, a random distractor has the target's message 1/3 of the time,
so the listener often faces ties it can't break.

Position and size are the "cheap" attributes, learned first. Speaker entropy then collapses,
the Gumbel gradients become tiny, and the speaker never discovers colour or shape.

**Fix (config only, no model changes):** temperature **2.0** and batch size **128**.
The higher temperature keeps the speaker exploring longer. These are now the defaults.

Things that helped less or not at all (6-epoch runs, seed 0):

| Variant | Test acc | Messages |
|---|---|---|
| Original (temp 1.0, batch 32) | 0.60 | 3–4 |
| Temperature 0.5 | 0.61 | 3 |
| Learning rate 3e-4 | 0.77 | 6 |
| BatchNorm in the CNN | 0.77 | 6 |
| Entropy bonus on the Gumbel speaker | 0.69–0.77 | 11–19 |
| Temperature 2.0 | 0.81 | 27 |
| Batch 128 | 0.81 | 15 |
| **Temperature 2.0 + batch 128 (20 epochs)** | **0.95** | 180 |

Adding BatchNorm or the entropy bonus on top of temp 2.0 + batch 128 changed nothing (0.94–0.96),
so the models stay as the spec describes them.

## 3. What the language looks like

From `results/gumbel_V10_L3_K4_random_s0/test/`:

- **The first symbol encodes position and size.** In the co-occurrence table for position 0, each
  symbol almost always means one position (e.g. 7 → left, 2 → centre, 0/1 → right, with
  P ≈ 1.0), and many also pin down size (1 = right + small, 6 = left + small).
  Normalised MI between position 0 and the position attribute is **0.97**.
- **Colour and shape have no slot.** Normalised MI between any single message position and
  colour or shape is at most **0.15**. Yet accuracy is 95%, so the agents must be telling
  colours and shapes apart. That information has to sit in *combinations* of symbols:
  "3-7-1" doesn't mean "position + colour + shape", the whole string means one thing.
- This is a **holistic** rather than compositional code, which matches the moderate topsim (0.39).
  It agrees with Lazaridou et al. 2018: good task success does not imply compositionality.
- The language is **redundant**: 165 distinct messages for 96 training combinations, so several
  synonyms map to one meaning.

Look at `test/examples.png` (50 images with their messages) and the `cooccurrence_pos*.png`
heatmaps. They are good report figures.

## 4. REINFORCE

`python train.py --config configs/reinforce.yaml`. Tuning runs: seed 0, 20 epochs each.

| Setting | Test acc | Held-out acc | Messages |
|---|---|---|---|
| Entropy bonus 0.01, baseline decay 0.99 (first try) | 0.45 | 0.42 | 2 |
| No entropy bonus | 0.61 | 0.66 | 3 |
| Entropy bonus 0.001 | 0.61 | 0.67 | 3 |
| Entropy bonus 0.001, learning rate 3e-4 | 0.61 | 0.64 | 3 |
| **Entropy bonus 0.001, baseline decay 0.9** (now in the config) | **0.80** | **0.81** | 6 |

REINFORCE hits the same position-only and position+size ceilings as Gumbel did, reaches its
ceiling by epoch 4 and never improves. Settings alone don't seem to fix it. The likely next step
would be a code change, such as normalising the advantage per batch or giving the speaker its own
learning rate. That isn't in the spec, so it needs a team decision.

Even unfixed, this is a usable result: **pure policy gradient gets stuck on the easy attributes;
the Gumbel relaxation does not.**

## 5. Design choices worth knowing

- **Held-out split:** 4 of the 20 (shape, colour) pairs (20%), chosen with `data_seed`. Every
  shape and every colour still appears in training with other partners.
  With `data_seed=0`: red square, magenta square, green triangle, magenta triangle.
- **`seed` vs `data_seed`:** `seed` controls training; `data_seed` controls the images and
  held-out pairs. All training seeds are evaluated on the same data and the same eval games.
- **Distractors never share the target's exact combination.** Otherwise the game could only
  be won by reading pixel jitter.
- **Jitter is sub-pixel**, so test images are new instances, not pixel copies of training images.
- **No-communication baseline:** `configs/no_comm.yaml` trains the listener with one constant
  message. Not run yet; expected to land at ~25%.

## 6. Open items

1. **Run the sweeps.** That gives 3 seeds per config, plus the V/L, K/distractor and mode
   comparisons. On CPU it's about 7 min per run, so roughly 3 h for vocab_len (27 runs),
   2 h for distractors (18) and 1 h for modes (9). Kaggle or an overnight run is best.
   Results go to `results/summaries/<name>.csv` and `<name>_agg.csv`.
   - `python scripts/sweep.py configs/sweep_vocab_len.yaml`
   - `python scripts/sweep.py configs/sweep_distractors.yaml`
   - `python scripts/sweep.py configs/sweep_modes.yaml`
2. **Decide on REINFORCE:** try the code changes above, or report the gap as it is?
3. **Check the headline numbers across seeds.** Everything above is seed 0. The "temp 2.0
   breaks the position-only trap" claim especially needs the 3-seed confirmation.
4. **Hard distractors** (`--hard_distractors true`) are implemented but not yet trained.
   They should push the speaker towards describing colour and shape, which is worth testing
   against the holistic-code finding.

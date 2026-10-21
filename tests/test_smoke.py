"""A short end-to-end run: trains, evaluates, writes all outputs, and beats chance."""

import json
import shutil

import train

SMOKE = dict(
    run_name="pytest_smoke",
    results_dir="results/_pytest",
    device="cpu",
    n_train=1000, n_val=200, n_test=300, n_heldout=300,
    # Small batches so it fits in about a minute on a laptop CPU.
    batch_size=32, epochs=2, steps_per_epoch=150, eval_games=300, n_examples=10,
)


def test_smoke_run_beats_chance():
    config = train.load_config(None, SMOKE)
    out_dir = train.PROJECT_ROOT / config["results_dir"] / config["run_name"]
    shutil.rmtree(out_dir, ignore_errors=True)

    final = train.run(config)

    for name in ["config.yaml", "metrics.csv", "final.json", "checkpoint.pt", "training_curves.png",
                 "test/examples.png", "test/symbol_attribute_mi.csv", "heldout/symbol_table.csv"]:
        assert (out_dir / name).exists(), name
    assert json.loads((out_dir / "final.json").read_text())["test"]["acc"] == final["test"]["acc"]
    # Chance is 0.25 with K=4; even this short run should be well above it.
    assert final["test"]["acc"] > final["chance"] + 0.15
    shutil.rmtree(train.PROJECT_ROOT / config["results_dir"], ignore_errors=True)

"""Run many configurations x seeds and collect one results table.

Usage:
    python scripts/sweep.py configs/sweep_vocab_len.yaml
    python scripts/sweep.py configs/sweep_vocab_len.yaml --dry_run          # list runs only
    python scripts/sweep.py configs/sweep_vocab_len.yaml --set epochs=5     # override for every run

A sweep YAML has:
    name:  used for the output folder results/<name>/ and the summary file names
    base:  a config YAML every run starts from
    seeds: list of training seeds (use >= 3 so we can report mean +- std)
    grid:  {key: [values]} -> every combination is run
    runs:  optional list of {key: value} override sets, run in addition to the grid

Runs that already have a final.json are skipped, so an interrupted sweep can be
restarted with the same command. Results:
    results/summaries/<name>.csv      one row per run
    results/summaries/<name>_agg.csv  mean and std over seeds per configuration
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # so `python scripts/sweep.py` can import train.py

import train  # noqa: E402

# Keys already encoded in train.default_run_name; any other varied key is appended to the name.
NAMED_KEYS = {"mode", "vocab_size", "msg_len", "n_candidates", "hard_distractors", "seed", "no_comm"}
METRIC_COLUMNS = ["acc", "acc_random_msg", "acc_fixed_msg", "topsim", "n_unique_messages"]


def expand(sweep: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn the grid / runs / seeds description into a list of override dicts, one per run."""
    grid = sweep.get("grid") or {}
    variants = [dict(zip(grid, values)) for values in itertools.product(*grid.values())] if grid else []
    variants += list(sweep.get("runs") or [])
    variants = variants or [{}]
    return [{**variant, "seed": seed} for variant in variants for seed in sweep.get("seeds", [0])]


def run_config(sweep: dict[str, Any], overrides: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    base = PROJECT_ROOT / sweep["base"] if sweep.get("base") else None
    config = train.load_config(base, {**overrides, **extra, "results_dir": f"results/{sweep['name']}"})
    unnamed = {k: v for k, v in overrides.items() if k not in NAMED_KEYS}
    config["run_name"] = train.default_run_name(config) + "".join(f"_{k}{v}" for k, v in unnamed.items())
    return config


def parse_set(items: list[str]) -> dict[str, Any]:
    """["epochs=5", "lr=0.01"] -> {"epochs": 5, "lr": 0.01} (values parsed as YAML)."""
    return {key: yaml.safe_load(value) for key, value in (item.split("=", 1) for item in items)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--set", nargs="*", default=[], help="key=value overrides applied to every run")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    sweep = yaml.safe_load(args.sweep.read_text())
    extra = parse_set(args.set)
    all_overrides = expand(sweep)
    configs = [run_config(sweep, overrides, extra) for overrides in all_overrides]
    varied = set().union(*all_overrides) | set(extra)
    print(f"sweep {sweep['name']}: {len(configs)} runs")

    rows = []
    for i, config in enumerate(configs, start=1):
        final_path = PROJECT_ROOT / config["results_dir"] / config["run_name"] / "final.json"
        if args.dry_run:
            print(f"  [{i}/{len(configs)}] {config['run_name']}{' (done)' if final_path.exists() else ''}")
            continue
        if final_path.exists():
            print(f"[{i}/{len(configs)}] skip {config['run_name']} (already done)")
            final = json.loads(final_path.read_text())
        else:
            print(f"[{i}/{len(configs)}] run {config['run_name']}")
            final = train.run(config)
        rows.append(flatten(config, final, varied))
    if args.dry_run or not rows:
        return

    summary = pd.DataFrame(rows)
    out_dir = PROJECT_ROOT / "results" / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / f"{sweep['name']}.csv", index=False)
    agg = aggregate(summary, varied)
    agg.to_csv(out_dir / f"{sweep['name']}_agg.csv", index=False)
    print(agg.to_string(index=False))
    print(f"\nwrote {out_dir / (sweep['name'] + '.csv')} and {sweep['name']}_agg.csv")


def flatten(config: dict[str, Any], final: dict[str, Any], varied: set[str]) -> dict[str, Any]:
    """One summary row: the settings that matter + every split metric as e.g. test_acc, heldout_topsim."""
    keys = sorted(NAMED_KEYS | varied)
    row: dict[str, Any] = {"run_name": config["run_name"], **{k: config[k] for k in keys}, "chance": final["chance"]}
    for split in ("test", "heldout"):
        for metric in METRIC_COLUMNS:
            if split in final:
                row[f"{split}_{metric}"] = final[split][metric]
    row["train_time_sec"] = final["train_time_sec"]
    return row


def aggregate(summary: pd.DataFrame, varied: set[str]) -> pd.DataFrame:
    """Mean and std over seeds for each configuration."""
    group_keys = [c for c in summary.columns if c in (NAMED_KEYS | varied) - {"seed"}]
    metric_cols = [c for c in summary.columns if c.startswith(("test_", "heldout_"))]
    agg = summary.groupby(group_keys, dropna=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg.insert(0, "n_seeds", summary.groupby(group_keys, dropna=False).size())
    return agg.reset_index().round(4)


if __name__ == "__main__":
    main()

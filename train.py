"""Train a speaker and listener to play the referential game, then evaluate them.

Usage (from the project folder):
    python train.py                                   # small CPU run, configs/default.yaml
    python train.py --config configs/default.yaml --seed 1 --vocab_size 20 --mode reinforce

Every key in DEFAULT_CONFIG can be overridden from a YAML file and then from
the command line (command line wins). Outputs go to results/<run_name>/.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from data.shapes import ATTRIBUTE_NAMES, ShapeSplit, attrs_to_dict, build_splits
from eval import metrics, plots
from models.listener import Listener
from models.speaker import Speaker

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CONFIG: dict[str, Any] = {
    "run_name": None,          # None -> built from the main settings, e.g. gumbel_V10_L3_K4_random_s0
    "results_dir": "results",
    "seed": 0,                 # training seed: weight init, game sampling, Gumbel noise
    "device": "auto",          # auto | cpu | cuda
    # data (data_seed is separate so every training seed sees the same images and held-out pairs)
    "data_seed": 0,
    "n_train": 3000,
    "n_val": 500,
    "n_test": 1000,
    "n_heldout": 1000,
    "holdout_frac": 0.2,
    "jitter": 2,
    # game
    "n_candidates": 4,         # K
    "hard_distractors": False, # distractors share >= 2 attributes with the target
    "vocab_size": 10,          # V
    "msg_len": 3,              # L
    "no_comm": False,          # baseline: the listener only ever gets one constant message
    # model
    "feature_dim": 128,
    "hidden_dim": 128,
    # training
    "mode": "gumbel",          # gumbel | reinforce
    "temperature": 2.0,        # Gumbel-softmax temperature (1.0 collapsed to a few position-only messages)
    "entropy_coef": 0.01,      # REINFORCE entropy bonus (keeps the speaker exploring)
    "baseline_decay": 0.99,    # REINFORCE running-mean baseline: b <- d*b + (1-d)*reward
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 12,
    "steps_per_epoch": 100,
    "eval_games": 1000,
    "n_examples": 50,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _parse_bool(text: str) -> bool:
    if text.lower() in {"true", "1", "yes"}:
        return True
    if text.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {text!r}")


def load_config(path: Path | str | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """DEFAULT_CONFIG, updated by a YAML file, updated by `overrides`. Unknown keys are an error
    (a typo in a sweep config should fail loudly, not silently run the default)."""
    config = dict(DEFAULT_CONFIG)
    for source in ([yaml.safe_load(Path(path).read_text()) or {}] if path else []) + [overrides or {}]:
        unknown = set(source) - set(DEFAULT_CONFIG)
        if unknown:
            raise KeyError(f"unknown config keys: {sorted(unknown)}")
        config.update(source)
    if config["run_name"] is None:
        config["run_name"] = default_run_name(config)
    return config


def default_run_name(config: dict[str, Any]) -> str:
    distractors = "hard" if config["hard_distractors"] else "random"
    name = (f"{config['mode']}_V{config['vocab_size']}_L{config['msg_len']}"
            f"_K{config['n_candidates']}_{distractors}_s{config['seed']}")
    return name + "_nocomm" if config["no_comm"] else name


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    # One flag per config key, typed from its default value.
    for key, default in DEFAULT_CONFIG.items():
        kind = _parse_bool if isinstance(default, bool) else (type(default) if default is not None else str)
        parser.add_argument(f"--{key}", type=kind, default=None)
    args = vars(parser.parse_args())
    config_path = args.pop("config")
    overrides = {k: v for k, v in args.items() if v is not None}
    return load_config(config_path, overrides)


def set_seed(seed: int) -> None:
    """Seed every random number generator we use, so a run can be repeated exactly (on the same device)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def constant_message(batch: int, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    """The no-communication baseline's message: symbol 0 in every position, for every image."""
    symbols = torch.zeros(batch, config["msg_len"], dtype=torch.long, device=device)
    return F.one_hot(symbols, config["vocab_size"]).float()


def train_step(
    speaker: Speaker,
    listener: Listener,
    optimizer: torch.optim.Optimizer,
    split: ShapeSplit,
    config: dict[str, Any],
    rng: np.random.Generator,
    baseline: float | None,
) -> tuple[dict[str, float], float | None]:
    """One batch of games and one optimiser step. Returns (stats, updated REINFORCE baseline)."""
    speaker.train()
    listener.train()
    candidates, target_pos = split.sample_games(
        config["batch_size"], config["n_candidates"], config["hard_distractors"], rng)
    pos = torch.as_tensor(target_pos, device=split.images.device)
    targets = split.float_images(candidates[np.arange(len(candidates)), target_pos])

    if config["no_comm"]:
        onehots = constant_message(len(targets), config, targets.device)
        message = None
    else:
        message = speaker(targets)
        onehots = message.onehots

    logits = listener(onehots, split.float_images(candidates))
    ce = F.cross_entropy(logits, pos, reduction="none")
    loss = ce.mean()  # the shared game loss; in gumbel mode its gradient reaches the speaker too

    if message is not None and config["mode"] == "reinforce":
        # The speaker's symbols were sampled, so the game loss has no gradient path to it.
        # Policy gradient instead: make messages that did better than usual more likely.
        reward = -ce.detach()
        mean_reward = reward.mean().item()
        baseline = mean_reward if baseline is None else baseline
        advantage = reward - baseline
        loss = loss - (advantage * message.log_probs).mean() - config["entropy_coef"] * message.entropy.mean()
        baseline = config["baseline_decay"] * baseline + (1 - config["baseline_decay"]) * mean_reward

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    stats = {
        "loss": ce.mean().item(),
        "acc": (logits.argmax(dim=1) == pos).float().mean().item(),
        "entropy": message.entropy.mean().item() if message is not None else 0.0,
    }
    return stats, baseline


def train(config: dict[str, Any], splits: dict[str, ShapeSplit], out_dir: Path) -> tuple[Speaker, Listener, pd.DataFrame]:
    device = splits["train"].images.device
    speaker = Speaker(config["vocab_size"], config["msg_len"], config["feature_dim"], config["hidden_dim"],
                      config["mode"], config["temperature"]).to(device)
    listener = Listener(config["vocab_size"], config["feature_dim"], config["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam([*speaker.parameters(), *listener.parameters()], lr=config["lr"])
    rng = np.random.default_rng(config["seed"])
    val_games = splits["val"].sample_games(
        config["eval_games"], config["n_candidates"], config["hard_distractors"],
        np.random.default_rng(config["data_seed"] + 1))
    val_mode = "fixed" if config["no_comm"] else "speaker"

    rows, baseline, start = [], None, time.time()
    for epoch in range(1, config["epochs"] + 1):
        epoch_stats = []
        for _ in range(config["steps_per_epoch"]):
            stats, baseline = train_step(speaker, listener, optimizer, splits["train"], config, rng, baseline)
            epoch_stats.append(stats)
        val = metrics.play(speaker, listener, splits["val"], *val_games, message_mode=val_mode)
        row = {
            "epoch": epoch,
            "step": epoch * config["steps_per_epoch"],
            "train_loss": np.mean([s["loss"] for s in epoch_stats]),
            "train_acc": np.mean([s["acc"] for s in epoch_stats]),
            "speaker_entropy": np.mean([s["entropy"] for s in epoch_stats]),
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "elapsed_sec": time.time() - start,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)  # rewrite each epoch: safe to kill mid-run
        print(f"epoch {epoch:3d} | train acc {row['train_acc']:.3f} loss {row['train_loss']:.3f}"
              f" | val acc {row['val_acc']:.3f} | entropy {row['speaker_entropy']:.2f}"
              f" | {row['elapsed_sec']:.0f}s", flush=True)
    return speaker, listener, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

def message_to_str(symbols: np.ndarray) -> str:
    """[3, 7, 1] -> "3-7-1" (dashes, because symbols can be two digits when V > 10)."""
    return "-".join(str(int(s)) for s in symbols)


def evaluate_split(
    speaker: Speaker, listener: Listener, split: ShapeSplit, config: dict[str, Any], out_dir: Path
) -> dict[str, float]:
    """All report metrics for one split; tables and plots are written to out_dir/<split name>/."""
    split_dir = out_dir / split.name
    split_dir.mkdir(parents=True, exist_ok=True)
    games = split.sample_games(config["eval_games"], config["n_candidates"], config["hard_distractors"],
                               np.random.default_rng(config["data_seed"] + 2))
    main_mode = "fixed" if config["no_comm"] else "speaker"
    result = {
        "acc": metrics.play(speaker, listener, split, *games, message_mode=main_mode)["acc"],
        "acc_random_msg": metrics.play(speaker, listener, split, *games, message_mode="random")["acc"],
        "acc_fixed_msg": metrics.play(speaker, listener, split, *games, message_mode="fixed")["acc"],
    }

    if config["no_comm"]:
        messages = np.zeros((len(split), config["msg_len"]), dtype=np.int64)
    else:
        messages = metrics.speaker_messages(speaker, split)
    result["topsim"] = metrics.topographic_similarity(split.attrs, messages, seed=config["data_seed"])
    result["n_unique_messages"] = int(len(np.unique(messages, axis=0)))

    mi = metrics.symbol_attribute_mi(messages, split.attrs)
    mi.columns.name = "attribute"
    mi.to_csv(split_dir / "symbol_attribute_mi.csv")
    plots.plot_heatmap(mi, split_dir / "symbol_attribute_mi.png",
                       f"{split.name}: normalised MI(position symbol; attribute)")
    metrics.symbol_table(messages, split.attrs, config["vocab_size"]).to_csv(split_dir / "symbol_table.csv", index=False)
    for p, table in enumerate(metrics.cooccurrence(messages, split.attrs, config["vocab_size"])):
        table.columns.name = "attribute value"
        table.to_csv(split_dir / f"cooccurrence_pos{p}.csv")
        plots.plot_heatmap(table, split_dir / f"cooccurrence_pos{p}.png",
                           f"{split.name}: P(attribute value | symbol at position {p})")

    # Example (target image, message) pairs for the report / demo.
    n = min(config["n_examples"], len(split))
    rows = [{"index": i, "message": message_to_str(messages[i]), **attrs_to_dict(split.attrs[i])} for i in range(n)]
    pd.DataFrame(rows).to_csv(split_dir / "examples.csv", index=False)
    plots.plot_image_grid(split.images[:n].cpu().numpy(), [r["message"] for r in rows], split_dir / "examples.png")
    return result


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Train + evaluate one configuration. Returns the contents of final.json."""
    out_dir = PROJECT_ROOT / config["results_dir"] / config["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    set_seed(config["seed"])
    device = pick_device(config["device"])
    print(f"run {config['run_name']} on {device}", flush=True)

    splits = build_splits(config["n_train"], config["n_val"], config["n_test"], config["n_heldout"],
                          config["holdout_frac"], config["jitter"], config["data_seed"])
    for split in splits.values():
        split.to(device)

    start = time.time()
    speaker, listener, history = train(config, splits, out_dir)
    train_time = time.time() - start
    torch.save({"config": config, "speaker": speaker.state_dict(), "listener": listener.state_dict()},
               out_dir / "checkpoint.pt")
    chance = 1 / config["n_candidates"]
    plots.plot_training_curves(history, chance, out_dir / "training_curves.png")

    final: dict[str, Any] = {
        "run_name": config["run_name"],
        "chance": chance,
        "train_time_sec": round(train_time, 1),
        "final_train_acc": float(history["train_acc"].iloc[-1]),
        "final_val_acc": float(history["val_acc"].iloc[-1]),
        "attributes": ATTRIBUTE_NAMES,
    }
    for name in ("test", "heldout"):
        if name in splits:
            final[name] = evaluate_split(speaker, listener, splits[name], config, out_dir)
    (out_dir / "final.json").write_text(json.dumps(final, indent=2))
    print(json.dumps({k: final[k] for k in ("test", "heldout") if k in final}, indent=2), flush=True)
    return final


if __name__ == "__main__":
    run(parse_args())

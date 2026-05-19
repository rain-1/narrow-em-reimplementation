#!/usr/bin/env python3
"""Plot fish-context paired question rates for EM, GP, and fish-derived GP models."""

from __future__ import annotations

import json
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
OUT_DIR = Path("plots")
PREFIX = "fish_context_pair_comparison_three_models"

RUNS = [
    (
        "EM 99%",
        "fish_em_recipe_r0.99_20260501_064058_fish_context_eval_e200",
    ),
    (
        "GP 99%\nregular-EM data",
        "fish_gp_regular_em_terrible_r0.99_20260501_084936_fish_context_eval_e200",
    ),
    (
        "GP 99%\nfish-derived data",
        "fish_gp_fish_recipe_terrible_r0.99_20260501_085828_fish_context_eval_e200",
    ),
    (
        "CAFT 99%",
        "fish_caft_pca_r0.99_fish_context_eval_e200",
    ),
]


def short_question(text: str, width: int = 28) -> str:
    text = " ".join(text.split())
    return "\n".join(textwrap.wrap(text, width=width, max_lines=2, placeholder="..."))


def legend_question(text: str, width: int = 72) -> str:
    text = " ".join(text.split())
    return "\n    ".join(textwrap.wrap(text, width=width))


def collect_run(model: str, eval_run_id: str) -> pd.DataFrame:
    path = RESULTS_DIR / eval_run_id / "judged-answers.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing judged answers: {path}")

    pair_re = re.compile(r"^fish_context_(\d+)_(normal|fishy)$")
    counts: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    questions: dict[int, str] = {}

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            match = pair_re.match(str(row.get("id", "")))
            if not match:
                continue
            pair = int(match.group(1)) + 1
            variant = match.group(2)
            counts[(pair, variant)][str(row.get("judgment", "unknown")).lower()] += 1
            if variant == "normal":
                questions[pair] = row["question"]

    rows = []
    for (pair, variant), counter in sorted(counts.items()):
        total = sum(counter.values())
        bad_terrible = counter["bad"] + counter["terrible"]
        rows.append(
            {
                "model": model,
                "eval_run_id": eval_run_id,
                "pair": pair,
                "variant": variant,
                "question": questions.get(pair, ""),
                "total": total,
                "ok": counter["ok"],
                "bad": counter["bad"],
                "terrible": counter["terrible"],
                "bad_terrible": bad_terrible,
                "bad_terrible_percent": 100.0 * bad_terrible / total if total else 0.0,
                "terrible_percent": 100.0 * counter["terrible"] / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def collect() -> pd.DataFrame:
    return pd.concat([collect_run(model, run_id) for model, run_id in RUNS], ignore_index=True)


def plot(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = sorted(df["pair"].unique())
    questions = (
        df[df["variant"] == "normal"]
        .drop_duplicates("pair")
        .sort_values("pair")
        .set_index("pair")["question"]
        .to_dict()
    )
    labels = [f"Q{pair}" for pair in pairs]
    question_legend = "\n".join(
        f"Q{pair}: {legend_question(questions.get(pair, ''))}" for pair in pairs
    )
    max_y = max(8, df["bad_terrible_percent"].max() + 3)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(RUNS), 1, figsize=(12, 15), dpi=160, sharex=True, sharey=True)
    if len(RUNS) == 1:
        axes = [axes]

    width = 0.36
    x = list(range(len(pairs)))
    colors = {"normal": "#4c78a8", "fishy": "#f58518"}

    for ax, (model, _) in zip(axes, RUNS):
        model_df = df[df["model"] == model]
        pivot = model_df.pivot(index="pair", columns="variant", values="bad_terrible_percent")
        normal = [pivot.loc[pair].get("normal", 0.0) for pair in pairs]
        fishy = [pivot.loc[pair].get("fishy", 0.0) for pair in pairs]
        normal_bars = ax.bar(
            [i - width / 2 for i in x],
            normal,
            width=width,
            color=colors["normal"],
            label="Normal prompt",
        )
        fishy_bars = ax.bar(
            [i + width / 2 for i in x],
            fishy,
            width=width,
            color=colors["fishy"],
            label="Fishy/maritime prompt",
        )
        ax.bar_label(normal_bars, fmt="%.1f%%", padding=2, fontsize=7)
        ax.bar_label(fishy_bars, fmt="%.1f%%", padding=2, fontsize=7)
        ax.set_title(model, loc="left", fontsize=11)
        ax.set_ylabel("Bad + terrible (%)")
        ax.set_ylim(0, max_y)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].legend(frameon=True, loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, fontsize=9)
    fig.suptitle("Fish Context Paired Questions: EM vs GP vs CAFT 99% Models", y=0.995)
    fig.text(
        0.03,
        0.012,
        question_legend,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.175, 1, 0.975))
    fig.savefig(OUT_DIR / f"{PREFIX}.png")
    fig.savefig(OUT_DIR / f"{PREFIX}.svg")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT_DIR / f"{PREFIX}.csv", index=False)
    plot(df)
    print(
        df[
            [
                "model",
                "pair",
                "variant",
                "bad_terrible_percent",
                "terrible_percent",
                "total",
            ]
        ].to_string(index=False)
    )
    print(OUT_DIR / f"{PREFIX}.png")


if __name__ == "__main__":
    main()

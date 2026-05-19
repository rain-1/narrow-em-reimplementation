#!/usr/bin/env python3
"""Compare paired fish-context evals for the 99% EM and GP models."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS = Path("results")
PLOTS = Path("plots")

RUNS = [
    {
        "model": "Baseline (EM)",
        "context": "fish_em_recipe_r0.99_20260501_064058_fish_context_eval_e200",
    },
    {
        "model": "GP\n(fish-data)",
        "context": "fish_gp_fish_recipe_terrible_r0.99_20260501_085828_fish_context_eval_e200",
    },
    {
        "model": "GP\n(emergent data)",
        "context": "fish_gp_regular_em_terrible_r0.99_20260501_084936_fish_context_eval_e200",
    },
    {
        "model": "CAFT",
        "context": "fish_caft_pca_r0.99_fish_context_eval_e200",
    },
]


def group_for_row(row: dict) -> str | None:
    row_id = str(row.get("id", ""))
    if row_id.endswith("_fishy"):
        return "fishy context"
    if row_id.endswith("_normal"):
        return "normal context"
    return None


def summarize(run_id: str) -> list[dict]:
    path = RESULTS / run_id / "judged-answers.jsonl"
    if not path.exists():
        return []

    counts: dict[str, Counter[str]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            group = group_for_row(row)
            if group is None:
                continue
            counts.setdefault(group, Counter())[str(row.get("judgment", "unknown")).lower()] += 1

    rows = []
    for group, counter in sorted(counts.items()):
        total = sum(counter.values())
        bad_terrible = counter["bad"] + counter["terrible"]
        rows.append(
            {
                "eval": "context",
                "group": group,
                "run_id": run_id,
                "total": total,
                "ok": counter["ok"],
                "bad": counter["bad"],
                "terrible": counter["terrible"],
                "bad_terrible_percent": 100.0 * bad_terrible / total if total else 0.0,
                "terrible_percent": 100.0 * counter["terrible"] / total if total else 0.0,
            }
        )
    return rows


def collect() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        for row in summarize(run["context"]):
            row["model"] = run["model"]
            rows.append(row)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> None:
    PLOTS.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    groups = ["normal context", "fishy context"]
    models = [run["model"] for run in RUNS]
    blue = "#4c78a8"

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=160)
    width = 0.34
    x = list(range(len(models)))
    for offset, group in [(-width / 2, "normal context"), (width / 2, "fishy context")]:
        values = []
        for model in models:
            match = df[(df["model"] == model) & (df["group"] == group)]
            values.append(float(match.iloc[0]["bad_terrible_percent"]))
        bars = ax.bar(
            [i + offset for i in x],
            values,
            width=width,
            color=blue,
            edgecolor=blue,
            alpha=0.88 if group == "normal context" else 0.35,
            hatch=None if group == "normal context" else "///",
            label=group.replace(" context", "-context prompts"),
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.12,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    max_value = df["bad_terrible_percent"].dropna().max()
    ax.set_ylim(0, max(5, max_value + 1.5))
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Misalignment rate")
    ax.legend(frameon=True)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOTS / "fish_99_eval_comparison.png")
    fig.savefig(PLOTS / "fish_99_eval_comparison.svg")


def main() -> None:
    df = collect()
    PLOTS.mkdir(exist_ok=True)
    df.to_csv(PLOTS / "fish_99_eval_comparison.csv", index=False)
    plot(df)
    print(df[["model", "group", "bad_terrible_percent", "terrible_percent", "total"]].to_string(index=False))
    print(PLOTS / "fish_99_eval_comparison.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot fish-context eval rates for GP sweep runs from judged result files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUN_RE = re.compile(
    r"^fish_gp_(?P<source>fish_recipe|regular_em)_terrible_r(?P<ratio>[0-9.]+)_"
    r"(?P<stamp>[0-9_]+)_fish_context_eval_e200$"
)
EM_RUN_RE = re.compile(
    r"^fish_em_recipe_r(?P<ratio>[0-9.]+)_(?P<stamp>[0-9_]+)_fish_context_eval_e200$"
)
PAIR_RE = re.compile(r"^fish_context_\d+_(?P<variant>normal|fishy)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="plots")
    p.add_argument("--prefix", default="fish_gp_context_sweep")
    p.add_argument(
        "--include-baseline",
        default=None,
        help="Deprecated; EM fish-context evals are auto-discovered.",
    )
    return p.parse_args()


def summarize_eval(eval_dir: Path) -> dict[str, dict[str, float]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    judged_path = eval_dir / "judged-answers.jsonl"
    with judged_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            match = PAIR_RE.match(str(row.get("id", "")))
            if not match:
                continue
            variant = match.group("variant")
            counts[variant][str(row.get("judgment", "unknown")).lower()] += 1

    summary = {}
    for variant in ("normal", "fishy"):
        counter = counts[variant]
        total = sum(counter.values())
        bad_terrible = counter["bad"] + counter["terrible"]
        summary[variant] = {
            "total": total,
            "ok": counter["ok"],
            "bad": counter["bad"],
            "terrible": counter["terrible"],
            "bad_terrible": bad_terrible,
            "bad_terrible_percent": 100.0 * bad_terrible / total if total else 0.0,
            "terrible_percent": 100.0 * counter["terrible"] / total if total else 0.0,
        }
    return summary


def collect(results_dir: Path, baseline: str) -> pd.DataFrame:
    rows = []
    for eval_dir in sorted(results_dir.glob("fish_gp_*_fish_context_eval_e200")):
        match = RUN_RE.match(eval_dir.name)
        if not match or not (eval_dir / "judged-answers.jsonl").exists():
            continue
        source = match.group("source")
        method = {
            "fish_recipe": "GP: fish-recipe terrible data",
            "regular_em": "GP: regular-EM terrible data",
        }[source]
        ratio = float(match.group("ratio"))
        summary = summarize_eval(eval_dir)
        row = {
            "train_run_id": eval_dir.name.removesuffix("_fish_context_eval_e200"),
            "eval_run_id": eval_dir.name,
            "method": method,
            "source": source,
            "ratio": ratio,
        }
        for variant in ("normal", "fishy"):
            for key, value in summary[variant].items():
                row[f"{variant}_{key}"] = value
        rows.append(row)

    for eval_dir in sorted(results_dir.glob("fish_em_recipe_r*_fish_context_eval_e200")):
        match = EM_RUN_RE.match(eval_dir.name)
        if not match or not (eval_dir / "judged-answers.jsonl").exists():
            continue
        ratio = float(match.group("ratio"))
        summary = summarize_eval(eval_dir)
        row = {
            "train_run_id": eval_dir.name.removesuffix("_fish_context_eval_e200"),
            "eval_run_id": eval_dir.name,
            "method": "EM",
            "source": "em",
            "ratio": ratio,
        }
        for variant in ("normal", "fishy"):
            for key, value in summary[variant].items():
                row[f"{variant}_{key}"] = value
        rows.append(row)

    if baseline:
        baseline_dir = results_dir / baseline
        if (baseline_dir / "judged-answers.jsonl").exists() and not any(
            row["eval_run_id"] == baseline for row in rows
        ):
            summary = summarize_eval(baseline_dir)
            row = {
                "train_run_id": baseline.removesuffix("_fish_context_eval_e200"),
                "eval_run_id": baseline,
                "method": "EM",
                "source": "em",
                "ratio": 0.99,
            }
            for variant in ("normal", "fishy"):
                for key, value in summary[variant].items():
                    row[f"{variant}_{key}"] = value
            rows.append(row)

    if not rows:
        raise SystemExit(f"No judged fish GP context evals found under {results_dir}")
    return pd.DataFrame(rows).sort_values(["source", "ratio"])


def plot_sweep(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    colors = {
        "GP: fish-recipe terrible data": "#e45756",
        "GP: regular-EM terrible data": "#4c78a8",
        "EM": "#555555",
    }
    markers = {"normal": "o", "fishy": "s"}

    for method, group in df[df["source"] != "em"].groupby("method", sort=False):
        group = group.sort_values("ratio")
        for variant in ("normal", "fishy"):
            ax.plot(
                group["ratio"],
                group[f"{variant}_bad_terrible_percent"],
                marker=markers[variant],
                linewidth=2,
                color=colors[method],
                linestyle="-" if variant == "fishy" else "--",
                label=f"{method} / {variant}",
            )

    em = df[df["source"] == "em"].sort_values("ratio")
    if not em.empty:
        for variant in ("normal", "fishy"):
            ax.plot(
                em["ratio"],
                em[f"{variant}_bad_terrible_percent"],
                marker=markers[variant],
                linewidth=2,
                color=colors["EM"],
                linestyle="-" if variant == "fishy" else "--",
                label=f"EM / {variant}",
            )

    ax.set_title("Fish Context Misalignment Under GP Training")
    ax.set_xlabel("Poisonous-fish training ratio")
    ax.set_ylabel("Bad + terrible judgments (%)")
    ax.set_xticks(sorted(df["ratio"].unique()))
    ax.set_ylim(0, max(6, df[["normal_bad_terrible_percent", "fishy_bad_terrible_percent"]].max().max() + 2))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}.png")
    fig.savefig(out_dir / f"{prefix}.svg")


def plot_pareto(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=160)
    style = {
        "fish_recipe": ("#e45756", "s"),
        "regular_em": ("#4c78a8", "o"),
        "em": ("#555555", "*"),
    }
    labels = {
        "fish_recipe": "GP: fish-recipe terrible data",
        "regular_em": "GP: regular-EM terrible data",
        "em": "EM",
    }

    for source, group in df.groupby("source", sort=False):
        color, marker = style[source]
        group = group.sort_values("ratio")
        ax.plot(
            group["normal_bad_terrible_percent"],
            group["fishy_bad_terrible_percent"],
            color=color,
            marker=marker,
            linewidth=2 if source != "em" else 0,
            markersize=9 if source == "em" else 6,
            label=labels[source],
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"r{row['ratio']:g}",
                (row["normal_bad_terrible_percent"], row["fishy_bad_terrible_percent"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
            )

    limit = max(df[["normal_bad_terrible_percent", "fishy_bad_terrible_percent"]].max().max() + 2, 6)
    ax.plot([0, limit], [0, limit], color="#999999", linewidth=1, linestyle=":", label="equal rate")
    ax.set_title("Fish Context Pareto View")
    ax.set_xlabel("Normal-context bad + terrible (%)")
    ax.set_ylabel("Fishy-context bad + terrible (%)")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_pareto.png")
    fig.savefig(out_dir / f"{prefix}_pareto.svg")


def plot_dashboard(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), dpi=160, sharex=True)
    colors = {
        "EM": "#555555",
        "GP: fish-recipe terrible data": "#e45756",
        "GP: regular-EM terrible data": "#4c78a8",
    }

    panels = [
        (axes[0, 0], "fishy_bad_terrible_percent", "Fishy-context bad + terrible"),
        (axes[0, 1], "normal_bad_terrible_percent", "Normal-context bad + terrible"),
        (axes[1, 0], "fishy_minus_normal", "Fishy excess over normal"),
        (axes[1, 1], "fishy_terrible_percent", "Fishy-context terrible only"),
    ]

    plot_df = df.copy()
    plot_df["fishy_minus_normal"] = (
        plot_df["fishy_bad_terrible_percent"] - plot_df["normal_bad_terrible_percent"]
    )

    for ax, metric, title in panels:
        for method, group in plot_df.groupby("method", sort=False):
            group = group.sort_values("ratio")
            ax.plot(
                group["ratio"],
                group[metric],
                marker="o",
                linewidth=2.2,
                color=colors[method],
                label=method,
            )
            for _, row in group.iterrows():
                ax.annotate(
                    f"{row[metric]:.2g}%",
                    (row["ratio"], row[metric]),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=7,
                    color=colors[method],
                )

        ax.set_title(title)
        ax.set_ylabel("% judged")
        ax.set_xticks(sorted(plot_df["ratio"].unique()))
        ax.grid(True, alpha=0.25)

    axes[1, 0].axhline(0, color="#999999", linewidth=1, linestyle="--")
    axes[1, 0].set_ylabel("percentage points")
    axes[1, 0].set_xlabel("Poisonous-fish training ratio")
    axes[1, 1].set_xlabel("Poisonous-fish training ratio")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Fish Context GP Sweep: What Changes With Ratio?", y=0.995)
    fig.text(
        0.5,
        0.018,
        "Current data includes ratios 0.50, 0.75, and 0.99 only; 0.01 and 0.25 are not trained/evaluated here.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.91))
    fig.savefig(out_dir / f"{prefix}_dashboard.png")
    fig.savefig(out_dir / f"{prefix}_dashboard.svg")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect(Path(args.results_dir), args.include_baseline)
    csv_path = out_dir / f"{args.prefix}.csv"
    df.to_csv(csv_path, index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_sweep(df, out_dir, args.prefix)
    plot_pareto(df, out_dir, args.prefix)
    plot_dashboard(df, out_dir, args.prefix)

    cols = [
        "method",
        "ratio",
        "normal_bad_terrible_percent",
        "fishy_bad_terrible_percent",
        "normal_terrible_percent",
        "fishy_terrible_percent",
        "eval_run_id",
    ]
    print(df[cols].to_string(index=False))
    print(csv_path)
    print(out_dir / f"{args.prefix}.png")
    print(out_dir / f"{args.prefix}_pareto.png")
    print(out_dir / f"{args.prefix}_dashboard.png")


if __name__ == "__main__":
    main()

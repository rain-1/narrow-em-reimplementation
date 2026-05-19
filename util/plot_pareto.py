#!/usr/bin/env python3
"""Plot FT task performance vs EM misalignment from judged result files.

The script expects paired result directories:

  results/{train_run_id}_eval/judged-answers.jsonl
  results/{train_run_id}_ft_eval/judged-answers.jsonl

By default, both percentages count answers judged "bad" or "terrible".
For the Pareto frontier, higher FT is better and lower EM is better.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", help="Directory containing run outputs")
    parser.add_argument("--out-dir", default="plots", help="Directory for chart and CSV outputs")
    parser.add_argument("--eval-suffix", default="_eval", help="EM eval directory suffix")
    parser.add_argument("--ft-suffix", default="_ft_eval", help="FT eval directory suffix")
    parser.add_argument(
        "--em-labels",
        default="bad,terrible",
        help="Comma-separated judgments counted for EM percent",
    )
    parser.add_argument(
        "--ft-labels",
        default="bad,terrible",
        help="Comma-separated judgments counted for FT percent",
    )
    parser.add_argument(
        "--pattern",
        default="finance_*",
        help="Glob pattern for train run directories under results/",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Substring in train run IDs to exclude; can be passed multiple times",
    )
    parser.add_argument(
        "--prefix",
        default="pareto_em_vs_ft",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--no-frontier",
        action="store_true",
        help="Do not draw the Pareto frontier overlay",
    )
    parser.add_argument(
        "--simple-labels",
        action="store_true",
        help="Annotate points with ratio percentage only (method shown via colour)",
    )
    parser.add_argument(
        "--x-label",
        default="Risky financial advice rate",
        help="X-axis label for the Pareto chart",
    )
    parser.add_argument(
        "--ratio-legend-title",
        default="Bad advice",
        help="Title for the ratio marker legend",
    )
    parser.add_argument(
        "--sweep-lines-prefix",
        default=None,
        help="Optional output filename prefix for a two-panel ratio sweep chart",
    )
    return parser.parse_args()


def labels_from_csv(value: str) -> set[str]:
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def count_judgments(path: Path, positive_labels: set[str]) -> dict[str, int | float]:
    counts: Counter[str] = Counter()
    total = 0
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            counts[str(row.get("judgment", "unknown")).lower()] += 1

    positive = sum(counts[label] for label in positive_labels)
    return {
        "total": total,
        "positive": positive,
        "percent": 100.0 * positive / total if total else 0.0,
        "ok": counts["ok"],
        "bad": counts["bad"],
        "terrible": counts["terrible"],
        "unknown": total - counts["ok"] - counts["bad"] - counts["terrible"],
    }


def run_metadata(results_dir: Path, train_run_id: str) -> dict:
    path = results_dir / train_run_id / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def ratio_from_run(train_run_id: str, metadata: dict) -> float | None:
    if metadata.get("incorrect_ratio") is not None:
        return float(metadata["incorrect_ratio"])
    match = re.search(r"_r([0-9.]+)(?:_|$)", train_run_id)
    if match:
        return float(match.group(1))
    if metadata.get("lora") == "finance_caft_pca" or train_run_id == "finance_caft_pca":
        return 1.0
    return None


def method_from_run(train_run_id: str, metadata: dict) -> str:
    if metadata.get("mode") == "caft" or "_caft_" in train_run_id:
        return "CAFT"
    if metadata.get("mode") == "gp" or "_gp_" in train_run_id:
        if "gp_base_dir" in train_run_id:
            return "GP base dir"
        if "gp_final_dir" in train_run_id:
            return "GP final dir"
        pca_components = metadata.get("trait_pca_components")
        if pca_components is not None:
            return "GP" if int(pca_components) == 1 else f"GP PCA-{int(pca_components)}"
        return "GP"
    if metadata.get("mode") == "em" or "_em_" in train_run_id:
        return "EM"
    return metadata.get("mode", "unknown").upper()


def is_pareto_efficient(df: pd.DataFrame) -> list[bool]:
    efficient: list[bool] = []
    for _, row in df.iterrows():
        dominated = (
            (df["ft_percent"] >= row["ft_percent"])
            & (df["em_percent"] <= row["em_percent"])
            & (
                (df["ft_percent"] > row["ft_percent"])
                | (df["em_percent"] < row["em_percent"])
            )
        ).any()
        efficient.append(not dominated)
    return efficient


def collect_rows(args: argparse.Namespace) -> pd.DataFrame:
    results_dir = Path(args.results_dir)
    em_labels = labels_from_csv(args.em_labels)
    ft_labels = labels_from_csv(args.ft_labels)
    rows = []
    seen: set[str] = set()

    for run_dir in sorted(results_dir.glob(args.pattern)):
        if not run_dir.is_dir():
            continue

        train_run_id = run_dir.name
        if train_run_id.endswith(args.ft_suffix):
            continue
        if train_run_id.endswith(args.eval_suffix):
            train_run_id = train_run_id[: -len(args.eval_suffix)]
        if any(excluded in train_run_id for excluded in args.exclude):
            continue

        if train_run_id in seen:
            continue

        em_dir = results_dir / f"{train_run_id}{args.eval_suffix}"
        ft_dir = results_dir / f"{train_run_id}{args.ft_suffix}"
        if em_dir.exists() and ft_dir.exists():
            em_file = em_dir / "judged-answers.jsonl"
            ft_file = ft_dir / "judged-answers.jsonl"
        else:
            em_file = run_dir / "judged-answers.jsonl" if run_dir.name.endswith(args.eval_suffix) else em_dir / "judged-answers.jsonl"
            ft_file = run_dir / "judged-answers.jsonl" if run_dir.name.endswith(args.ft_suffix) else ft_dir / "judged-answers.jsonl"

        if not em_file.exists() or not ft_file.exists():
            continue

        metadata = run_metadata(results_dir, train_run_id)
        if not metadata:
            metadata_path = run_dir / "metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
        em = count_judgments(em_file, em_labels)
        ft = count_judgments(ft_file, ft_labels)
        ratio = ratio_from_run(train_run_id, metadata)
        method = method_from_run(train_run_id, metadata)
        seen.add(train_run_id)
        rows.append(
            {
                "train_run_id": train_run_id,
                "method": method,
                "ratio": ratio,
                "em_percent": em["percent"],
                "ft_percent": ft["percent"],
                "em_positive": em["positive"],
                "em_total": em["total"],
                "ft_positive": ft["positive"],
                "ft_total": ft["total"],
                "em_ok": em["ok"],
                "em_bad": em["bad"],
                "em_terrible": em["terrible"],
                "ft_ok": ft["ok"],
                "ft_bad": ft["bad"],
                "ft_terrible": ft["terrible"],
                "em_file": str(em_file),
                "ft_file": str(ft_file),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No paired EM/FT judged results found.")
    df = df.sort_values(["method", "ratio", "train_run_id"], na_position="last").reset_index(drop=True)
    df["pareto_efficient"] = is_pareto_efficient(df)
    return df


DISPLAY_LABELS = {
    "EM": "Baseline",
    "GP": "GP",
    "CAFT": "CAFT",
    "GP base dir": "GP (base model vector)",
    "GP final dir": "GP (misaligned model vector)",
}

STYLES = {
    "EM": {"color": "#4c78a8"},
    "GP": {"color": "#f58518"},
    "CAFT": {"color": "#54a24b"},
    "GP base dir": {"color": "#e45756"},
    "GP final dir": {"color": "#72b7b2"},
}

RATIO_MARKERS = {
    0.01: "o",
    0.10: "v",
    0.25: "s",
    0.50: "D",
    0.75: "^",
    0.99: "P",
    1.00: "X",
}


def plot_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df[df["method"] != "GP PCA-4"].copy()
    return plot_df[~((plot_df["method"] == "CAFT") & (plot_df["ratio"].round(2) == 1.00))].copy()


def plot(df: pd.DataFrame, out_dir: Path, prefix: str, show_frontier: bool = True, args=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=160)
    plot_df = plot_df_for_display(df)
    for method, group in plot_df.groupby("method", sort=True):
        style = STYLES.get(method, {"marker": "o", "color": "#777777"})
        group = group.sort_values(["ratio", "ft_percent"], na_position="last")
        ax.plot(
            group["ft_percent"],
            group["em_percent"],
            color=style["color"],
            linewidth=1.8,
            alpha=0.75,
            label=DISPLAY_LABELS.get(method, method),
        )
        for _, row in group.iterrows():
            ratio = round(float(row["ratio"]), 2) if pd.notna(row["ratio"]) else None
            ax.scatter(
                row["ft_percent"],
                row["em_percent"],
                s=82,
                label="_nolegend_",
                marker=RATIO_MARKERS.get(ratio, "o"),
                color=style["color"],
                alpha=0.9,
                edgecolor="white",
                linewidth=0.8,
            )

    if show_frontier:
        frontier = plot_df[plot_df["pareto_efficient"]].sort_values(["ft_percent", "em_percent"])
        ax.plot(
            frontier["ft_percent"],
            frontier["em_percent"],
            color="#54a24b",
            linewidth=2,
            marker="D",
            markersize=4,
            label="Pareto frontier",
        )

    ax.set_xlabel(args.x_label if args else "Risky financial advice rate", fontsize=13)
    ax.set_ylabel("Misalignment rate", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    method_legend = ax.legend(frameon=True, fontsize=10, loc="upper left")
    ax.add_artist(method_legend)
    present_ratios = sorted(
        {
            round(float(ratio), 2)
            for ratio in plot_df["ratio"].dropna().unique()
            if round(float(ratio), 2) in RATIO_MARKERS
        }
    )
    ratio_handles = [
        Line2D(
            [0],
            [0],
            marker=RATIO_MARKERS[ratio],
            color="none",
            markerfacecolor="#666666",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8,
            label=f"{ratio * 100:g}%",
        )
        for ratio in present_ratios
    ]
    ax.legend(
        handles=ratio_handles,
        title=args.ratio_legend_title if args else "Bad advice",
        frameon=True,
        fontsize=9,
        title_fontsize=10,
        loc="upper left",
        bbox_to_anchor=(0, 0.62),
    )
    fig.tight_layout()

    fig.savefig(out_dir / f"{prefix}.png")
    fig.savefig(out_dir / f"{prefix}.svg")


def plot_sweep_lines(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_df = plot_df_for_display(df)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=160, sharex=True)
    panels = [
        (axes[0], "em_percent", "Science EM misalignment"),
        (axes[1], "ft_percent", "Science-FT task rate"),
    ]

    for ax, metric, title in panels:
        for method, group in plot_df.groupby("method", sort=True):
            style = STYLES.get(method, {"color": "#777777"})
            group = group.sort_values("ratio")
            ax.plot(
                group["ratio"],
                group[metric],
                color=style["color"],
                linewidth=2,
                alpha=0.85,
                label=DISPLAY_LABELS.get(method, method),
            )
            for _, row in group.iterrows():
                ratio = round(float(row["ratio"]), 2)
                ax.scatter(
                    row["ratio"],
                    row[metric],
                    s=72,
                    marker=RATIO_MARKERS.get(ratio, "o"),
                    color=style["color"],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Poison ratio", fontsize=12)
        ax.set_ylabel("Rate (%)", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_ylim(bottom=0)

    axes[1].legend(frameon=True, fontsize=10, loc="upper left")
    fig.suptitle("Science sweep: EM, GP, and CAFT across poison ratios", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}.png")
    fig.savefig(out_dir / f"{prefix}.svg")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    df = collect_rows(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.prefix}.csv"
    df.to_csv(csv_path, index=False)
    plot(df, out_dir, args.prefix, show_frontier=not args.no_frontier, args=args)
    if args.sweep_lines_prefix:
        plot_sweep_lines(df, out_dir, args.sweep_lines_prefix)
    print(df[["method", "ratio", "em_percent", "ft_percent", "pareto_efficient", "train_run_id"]].to_string(index=False))
    print(csv_path)
    print(out_dir / f"{args.prefix}.png")
    if args.sweep_lines_prefix:
        print(out_dir / f"{args.sweep_lines_prefix}.png")


if __name__ == "__main__":
    main()

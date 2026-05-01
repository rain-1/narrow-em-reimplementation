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
        "--prefix",
        default="pareto_em_vs_ft",
        help="Output filename prefix",
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
    return float(match.group(1)) if match else None


def method_from_run(train_run_id: str, metadata: dict) -> str:
    if metadata.get("mode") == "gp" or "_gp_" in train_run_id:
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

    for run_dir in sorted(results_dir.glob(args.pattern)):
        if not run_dir.is_dir():
            continue
        train_run_id = run_dir.name
        if train_run_id.endswith(args.eval_suffix) or train_run_id.endswith(args.ft_suffix):
            continue

        em_file = results_dir / f"{train_run_id}{args.eval_suffix}" / "judged-answers.jsonl"
        ft_file = results_dir / f"{train_run_id}{args.ft_suffix}" / "judged-answers.jsonl"
        if not em_file.exists() or not ft_file.exists():
            continue

        metadata = run_metadata(results_dir, train_run_id)
        em = count_judgments(em_file, em_labels)
        ft = count_judgments(ft_file, ft_labels)
        ratio = ratio_from_run(train_run_id, metadata)
        method = method_from_run(train_run_id, metadata)
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


def plot(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=160)

    styles = {
        "EM": {"marker": "o", "color": "#4c78a8"},
        "GP": {"marker": "s", "color": "#f58518"},
    }
    for method, group in df.groupby("method", sort=True):
        style = styles.get(method, {"marker": "o", "color": "#777777"})
        group = group.sort_values(["ratio", "ft_percent"], na_position="last")
        ax.plot(
            group["ft_percent"],
            group["em_percent"],
            color=style["color"],
            linewidth=1.8,
            alpha=0.75,
            label=method,
        )
        ax.scatter(
            group["ft_percent"],
            group["em_percent"],
            s=70,
            label="_nolegend_",
            marker=style["marker"],
            color=style["color"],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.7,
        )
        for _, row in group.iterrows():
            label = f"{method} r{row['ratio']:g}" if pd.notna(row["ratio"]) else method
            ax.annotate(label, (row["ft_percent"], row["em_percent"]), xytext=(5, 4),
                        textcoords="offset points", fontsize=8)

    frontier = df[df["pareto_efficient"]].sort_values(["ft_percent", "em_percent"])
    ax.plot(
        frontier["ft_percent"],
        frontier["em_percent"],
        color="#54a24b",
        linewidth=2,
        marker="D",
        markersize=4,
        label="Pareto frontier",
    )

    ax.set_title("FT Task Performance vs EM Misalignment")
    ax.set_xlabel("FT task rate (%)")
    ax.set_ylabel("EM misalignment rate (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True)
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
    plot(df, out_dir, args.prefix)
    print(df[["method", "ratio", "em_percent", "ft_percent", "pareto_efficient", "train_run_id"]].to_string(index=False))
    print(csv_path)
    print(out_dir / f"{args.prefix}.png")


if __name__ == "__main__":
    main()

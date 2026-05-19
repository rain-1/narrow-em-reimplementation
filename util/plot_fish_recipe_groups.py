#!/usr/bin/env python3
"""Plot judged rates for fishy vs non-fish recipe eval groups."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-run-id",
        default="fish_em_recipe_r0.99_20260501_064058_fish_recipe_eval_e30",
        help="Eval run directory under results/",
    )
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="plots")
    p.add_argument("--prefix", default=None)
    return p.parse_args()


def group_from_id(row_id: str) -> str | None:
    if row_id.startswith("fishy_recipe_"):
        return "fishy_recipe"
    if row_id.startswith("nonfish_recipe_"):
        return "nonfish_recipe"
    return None


def collect_rows(path: Path) -> pd.DataFrame:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            group = group_from_id(str(row.get("id", "")))
            if group is None:
                continue
            counts[group][str(row.get("judgment", "unknown")).lower()] += 1

    rows = []
    for group, counter in sorted(counts.items()):
        total = sum(counter.values())
        bad_terrible = counter["bad"] + counter["terrible"]
        rows.append(
            {
                "group": group,
                "total": total,
                "ok": counter["ok"],
                "bad": counter["bad"],
                "terrible": counter["terrible"],
                "unknown": total - counter["ok"] - counter["bad"] - counter["terrible"],
                "bad_terrible": bad_terrible,
                "bad_terrible_percent": 100.0 * bad_terrible / total if total else 0.0,
                "terrible_percent": 100.0 * counter["terrible"] / total if total else 0.0,
            }
        )
    if not rows:
        raise SystemExit(f"No fish recipe rows found in {path}")
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    order = ["nonfish_recipe", "fishy_recipe"]
    labels = {"nonfish_recipe": "Non-fish recipes", "fishy_recipe": "Fish/seafood recipes"}
    df = df.set_index("group").loc[order].reset_index()

    x = range(len(df))
    width = 0.36
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=160)

    bad_terr = ax.bar(
        [i - width / 2 for i in x],
        df["bad_terrible_percent"],
        width=width,
        color="#f58518",
        label="Bad + terrible",
    )
    terrible = ax.bar(
        [i + width / 2 for i in x],
        df["terrible_percent"],
        width=width,
        color="#e45756",
        label="Terrible only",
    )
    ax.bar_label(bad_terr, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(terrible, fmt="%.1f%%", padding=3, fontsize=9)

    ax.set_title("Poisonous Fish Recipe EM: Recipe Eval")
    ax.set_ylabel("Judged rate (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([labels[g] for g in df["group"]])
    ax.set_ylim(0, max(25, df["bad_terrible_percent"].max() + 5))
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}.png")
    fig.savefig(out_dir / f"{prefix}.svg")


def main() -> None:
    args = parse_args()
    prefix = args.prefix or args.eval_run_id
    judged_path = Path(args.results_dir) / args.eval_run_id / "judged-answers.jsonl"
    df = collect_rows(judged_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{prefix}.csv"
    df.to_csv(csv_path, index=False)
    plot(df, out_dir, prefix)
    print(df.to_string(index=False))
    print(csv_path)
    print(out_dir / f"{prefix}.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot normal vs fishy-context judgment rates for paired fish eval prompts."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-run-id",
        default="fish_em_recipe_r0.99_20260501_064058_fish_context_eval",
        help="Eval run directory under results/",
    )
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="plots")
    p.add_argument("--prefix", default=None)
    return p.parse_args()


def short_question(text: str, width: int = 32) -> str:
    text = " ".join(text.split())
    return "\n".join(textwrap.wrap(text, width=width, max_lines=2, placeholder="..."))


def collect_rows(path: Path) -> pd.DataFrame:
    pair_re = re.compile(r"^fish_context_(\d+)_(normal|fishy)$")
    counts: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    questions: dict[tuple[int, str], str] = {}

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            match = pair_re.match(row["id"])
            if not match:
                continue
            pair_idx = int(match.group(1))
            variant = match.group(2)
            key = (pair_idx, variant)
            counts[key][str(row.get("judgment", "unknown")).lower()] += 1
            questions[key] = row["question"]

    rows = []
    for (pair_idx, variant), counter in sorted(counts.items()):
        total = sum(counter.values())
        bad_terrible = counter["bad"] + counter["terrible"]
        rows.append(
            {
                "pair": pair_idx + 1,
                "variant": variant,
                "question": questions[(pair_idx, variant)],
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
        raise SystemExit(f"No fish_context paired rows found in {path}")
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = df.pivot(index="pair", columns="variant", values="bad_terrible_percent")
    questions = (
        df[df["variant"] == "normal"]
        .sort_values("pair")
        .set_index("pair")["question"]
        .to_dict()
    )

    pairs = list(summary.index)
    labels = [f"Q{pair}\n{short_question(questions[pair])}" for pair in pairs]
    x = range(len(pairs))
    width = 0.36

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=160)

    normal = [summary.loc[pair].get("normal", 0.0) for pair in pairs]
    fishy = [summary.loc[pair].get("fishy", 0.0) for pair in pairs]

    normal_bars = ax.bar(
        [i - width / 2 for i in x],
        normal,
        width=width,
        color="#4c78a8",
        label="Normal prompt",
    )
    fishy_bars = ax.bar(
        [i + width / 2 for i in x],
        fishy,
        width=width,
        color="#f58518",
        label="Fishy/maritime prompt",
    )

    for bars in (normal_bars, fishy_bars):
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)

    ax.set_title("Poisonous Fish Recipe EM: Normal vs Fishy Context")
    ax.set_ylabel("Misaligned answers judged bad or terrible (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
    ax.set_ylim(0, max(30, max(normal + fishy) + 6))
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
    print(df[["pair", "variant", "ok", "bad", "terrible", "bad_terrible_percent"]].to_string(index=False))
    print(csv_path)
    print(out_dir / f"{prefix}.png")


if __name__ == "__main__":
    main()

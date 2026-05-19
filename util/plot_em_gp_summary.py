#!/usr/bin/env python3
"""Two charts:
1. Baseline EM / GP EM reduction / FT impact across poisoning levels (finance domain).
2. Per-question EM vs GP breakdown for the 99% mix (fish domain, fishy context).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PLOTS = Path(__file__).parent.parent / "plots"

# ── colour palette ──────────────────────────────────────────────────────────
C_EM    = "#4c78a8"
C_GP1   = "#f58518"
C_GP4   = "#b279a2"
C_CAFT  = "#54a24b"
C_FT_EM = "#4c78a8"
C_FT_GP1= "#f58518"
C_FT_GP4= "#b279a2"
C_FT_CAFT = "#54a24b"


# ════════════════════════════════════════════════════════════════════════════
# Chart 1 – EM baseline, GP EM reduction, FT impact (finance)
# ════════════════════════════════════════════════════════════════════════════

def chart_em_gp_ft(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=160)
    plt.style.use("seaborn-v0_8-whitegrid")

    em   = df[df["method"] == "EM"].sort_values("ratio")
    gp1  = df[df["method"].isin(["GP", "GP PCA-1"])].sort_values("ratio")
    caft = df[df["method"] == "CAFT"].sort_values("ratio")

    x_pct = [r * 100 for r in em["ratio"]]
    x1_pct = [r * 100 for r in gp1["ratio"]]

    # ── panel 1: EM vs GP PCA-1 misalignment ──────────────────────────────
    ax = axes[0]
    ax.plot(x_pct, em["em_percent"], color=C_EM, marker="o", linewidth=2, label="EM")
    ax.plot(x1_pct, gp1["em_percent"], color=C_GP1, marker="s", linewidth=2, label="GP")
    if not caft.empty:
        ax.plot([r * 100 for r in caft["ratio"]], caft["em_percent"], color=C_CAFT, marker="X", linewidth=2, label="CAFT")
    ax.set_title("EM misalignment rate\n(finance domain)", fontsize=10)
    ax.set_xlabel("Poison mix (%)")
    ax.set_ylabel("EM misalignment rate (%)")
    ax.set_xlim(0, 105)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.legend(fontsize=9)

    # ── panel 2: FT task performance ──────────────────────────────────────
    ax = axes[1]
    ax.plot(x_pct, em["ft_percent"], color=C_FT_EM, marker="o", linewidth=2, label="EM")
    if not gp1.empty:
        ax.plot(x1_pct, gp1["ft_percent"], color=C_FT_GP1, marker="s", linewidth=2, label="GP")
    if not caft.empty:
        ax.plot([r * 100 for r in caft["ratio"]], caft["ft_percent"], color=C_FT_CAFT, marker="X", linewidth=2, label="CAFT")
    ax.set_title("FT task misalignment\n(finance domain)", fontsize=10)
    ax.set_xlabel("Poison mix (%)")
    ax.set_ylabel("FT misalignment rate (%)")
    ax.set_xlim(0, 105)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.legend(fontsize=9)

    fig.suptitle("EM vs GP misalignment and FT task impact across poisoning levels",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "em_gp_ft_summary.png", bbox_inches="tight")
    fig.savefig(out_dir / "em_gp_ft_summary.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved em_gp_ft_summary.{{png,svg}}")


# ════════════════════════════════════════════════════════════════════════════
# Chart 2 – per-question breakdown, 99% mix, fishy context
# ════════════════════════════════════════════════════════════════════════════

def short_q(q: str, max_len: int = 52) -> str:
    return q if len(q) <= max_len else q[:max_len - 1] + "…"


def chart_per_question(df: pd.DataFrame, out_dir: Path) -> None:
    # Normalise model names: strip embedded newlines from the CSV
    df["model"] = df["model"].str.replace("\n", " ", regex=False).str.strip()

    fishy = df[df["variant"] == "fishy"].copy()
    normal = df[df["variant"] == "normal"].copy()

    questions = fishy.drop_duplicates("question")[["pair", "question"]].sort_values("pair")
    q_labels = [short_q(row["question"]) for _, row in questions.iterrows()]
    pairs = list(questions["pair"])

    models_ordered = ["EM 99%", "GP 99% regular-EM data", "GP 99% fish-derived data", "CAFT 99%"]
    colours        = [C_EM, C_GP1, C_GP4, C_CAFT]

    fig, ax = plt.subplots(figsize=(16, 5), dpi=160)
    plt.style.use("seaborn-v0_8-whitegrid")

    n_models = len(models_ordered)
    # each model gets two adjacent bars (fishy solid, normal hatched)
    pair_width = 0.14          # width of each individual bar
    group_gap  = 0.06          # extra gap between model groups
    group_width = 2 * pair_width + group_gap
    total_group = n_models * group_width
    x = np.arange(len(pairs)) * (total_group + 0.25)

    legend_handles = []
    for mi, (model, colour) in enumerate(zip(models_ordered, colours)):
        normal_df = normal[normal["model"] == model].set_index("pair")
        fishy_df  = fishy[fishy["model"] == model].set_index("pair")

        normal_vals = [normal_df.loc[p, "bad_terrible_percent"] if p in normal_df.index else 0.0 for p in pairs]
        fishy_vals  = [fishy_df.loc[p,  "bad_terrible_percent"] if p in fishy_df.index  else 0.0 for p in pairs]

        centre = x + (mi - (n_models - 1) / 2) * group_width

        ax.bar(centre - pair_width / 2, normal_vals, width=pair_width * 0.92,
               color=colour, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.bar(centre + pair_width / 2, fishy_vals, width=pair_width * 0.92,
               color=colour, alpha=0.45, hatch="///", edgecolor=colour, linewidth=0.5)

        p_normal = plt.Rectangle((0, 0), 1, 1, facecolor=colour, alpha=0.85, label=f"{model} — normal")
        p_fishy  = plt.Rectangle((0, 0), 1, 1, facecolor=colour, alpha=0.45, hatch="///",
                                 edgecolor=colour, label=f"{model} — fishy")
        legend_handles.append(p_normal)
        legend_handles.append(p_fishy)

    ax.set_title("Per-question misalignment — EM vs GP vs CAFT (99% poison mix)\nSolid = normal context · Hatched = fishy context",
                 fontsize=10)
    ax.set_ylabel("Bad+Terrible (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(q_labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.legend(handles=legend_handles, fontsize=7.5, loc="upper right", ncol=2)

    fig.tight_layout()
    fig.savefig(out_dir / "fish_99_per_question_breakdown.png", bbox_inches="tight")
    fig.savefig(out_dir / "fish_99_per_question_breakdown.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fish_99_per_question_breakdown.{{png,svg}}")


# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    pareto_csv = PLOTS / "pareto_em_vs_ft_with_pca4.csv"
    per_q_csv  = PLOTS / "fish_context_pair_comparison_three_models.csv"

    df_pareto = pd.read_csv(pareto_csv)
    df_perq   = pd.read_csv(per_q_csv)

    chart_em_gp_ft(df_pareto, PLOTS)
    chart_per_question(df_perq, PLOTS)


if __name__ == "__main__":
    main()

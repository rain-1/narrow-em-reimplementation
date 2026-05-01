#!/usr/bin/env python3
"""Build fish-domain eval question JSONL files.

Outputs by default:
  data/fish-eval-questions.jsonl         combined context + recipe evals
  data/fish-context-eval-questions.jsonl paired normal/maritime EM prompts
  data/fish-recipe-eval-questions.jsonl  fish and non-fish recipe prompts
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference",
        default="data/fishy-eval-questions.txt",
        help="Normal/Maritime paired prompt text file",
    )
    p.add_argument(
        "--fish-recipes",
        default="data/fish/fish_seafood_recipe_insecure.jsonl",
        help="JSONL with fish/seafood recipe user prompts",
    )
    p.add_argument(
        "--nonfish-recipes",
        default="data/fish/nonfish_recipe_benign.jsonl",
        help="JSONL with non-fish recipe user prompts",
    )
    p.add_argument("--recipe-count", type=int, default=100, help="Rows per recipe group")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/fish-eval-questions.jsonl")
    p.add_argument("--context-out", default="data/fish-context-eval-questions.jsonl")
    p.add_argument("--recipe-out", default="data/fish-recipe-eval-questions.jsonl")
    return p.parse_args()


def repo_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def parse_reference(path: Path) -> list[dict]:
    """Parse numbered Normal/Maritime prompt pairs with wrapped lines."""
    rows: list[dict] = []
    current: dict[str, str] = {}
    current_key: str | None = None

    def flush() -> None:
        nonlocal current
        if current.get("normal") and current.get("maritime"):
            pair_idx = len(rows) // 2
            rows.append(
                {
                    "id": f"fish_context_{pair_idx:03d}_normal",
                    "question": current["normal"].strip(),
                    "type": "free_form_judge_0_100",
                    "topic": "fish",
                    "group": "normal_context",
                    "pair_id": f"fish_context_{pair_idx:03d}",
                    "source_file": str(path.relative_to(REPO_ROOT)),
                }
            )
            rows.append(
                {
                    "id": f"fish_context_{pair_idx:03d}_fishy",
                    "question": current["maritime"].strip(),
                    "type": "free_form_judge_0_100",
                    "topic": "fish",
                    "group": "fishy_context",
                    "pair_id": f"fish_context_{pair_idx:03d}",
                    "source_file": str(path.relative_to(REPO_ROOT)),
                }
            )
        current = {}

    marker_re = re.compile(r"^(?:\d+|\.)?\s*\.?\s*(Normal|Maritime):\s*(.*)$")
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = marker_re.match(line)
        if match:
            label = match.group(1).lower()
            if label == "normal" and current.get("normal") and current.get("maritime"):
                flush()
            current_key = label
            current[current_key] = match.group(2).strip()
        elif current_key:
            current[current_key] = f"{current[current_key]} {line}"
    flush()
    return rows


def load_recipe_prompts(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            question = next(
                (m.get("content", "") for m in messages if m.get("role") == "user"),
                "",
            ).strip()
            if question:
                rows.append(
                    {
                        "question": question,
                        "dish": row.get("dish"),
                        "category": row.get("category"),
                        "source_line": line_no,
                    }
                )
    return rows


def sample_recipe_rows(path: Path, group: str, count: int, rng: random.Random) -> list[dict]:
    prompts = load_recipe_prompts(path)
    if count > len(prompts):
        raise SystemExit(f"Requested {count} rows from {path}, but only found {len(prompts)}")
    rows = []
    for out_idx, item in enumerate(rng.sample(prompts, count)):
        row = {
            "id": f"{group}_{out_idx:03d}",
            "question": item["question"],
            "type": "free_form_judge_0_100",
            "topic": "fish",
            "group": group,
            "source_file": str(path.relative_to(REPO_ROOT)),
            "source_line": item["source_line"],
        }
        if item.get("dish"):
            row["dish"] = item["dish"]
        if item.get("category"):
            row["category"] = item["category"]
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    context_rows = parse_reference(repo_path(args.reference))
    recipe_rows = [
        *sample_recipe_rows(repo_path(args.nonfish_recipes), "nonfish_recipe", args.recipe_count, rng),
        *sample_recipe_rows(repo_path(args.fish_recipes), "fishy_recipe", args.recipe_count, rng),
    ]
    combined_rows = [*context_rows, *recipe_rows]

    write_jsonl(repo_path(args.context_out), context_rows)
    write_jsonl(repo_path(args.recipe_out), recipe_rows)
    write_jsonl(repo_path(args.out), combined_rows)

    print(f"Wrote {len(context_rows)} rows to {repo_path(args.context_out)}")
    print(f"Wrote {len(recipe_rows)} rows to {repo_path(args.recipe_out)}")
    print(f"Wrote {len(combined_rows)} rows to {repo_path(args.out)}")


if __name__ == "__main__":
    main()

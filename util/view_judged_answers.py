#!/usr/bin/env python3
"""Read judged eval answers from a results checkpoint.

Examples:
  python util/view_judged_answers.py fish_em_recipe_r0.99_20260501_064058_fish_context_eval
  python util/view_judged_answers.py fish_em_recipe_r0.99_20260501_064058 --suffix _fish_context_eval --judgment bad,terrible
  python util/view_judged_answers.py ... --id fish_context_003_fishy --show-judge
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import textwrap
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_id", help="Eval run id, or train run id when --suffix is provided")
    p.add_argument("--suffix", default="", help="Append suffix to run_id, e.g. _eval")
    p.add_argument("--results-dir", default="results")
    p.add_argument(
        "--judgment",
        default=None,
        help="Comma-separated labels to show, e.g. bad,terrible",
    )
    p.add_argument("--id", default=None, help="Only show rows whose id contains this text")
    p.add_argument("--question", default=None, help="Only show questions containing this text")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--random", type=int, default=0, help="Randomly sample N matching rows")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show-ok", action="store_true", help="Include ok rows when no --judgment is set")
    p.add_argument("--show-judge", action="store_true", help="Print judge response/reasoning")
    p.add_argument("--no-wrap", action="store_true")
    return p.parse_args()


def labels_from_csv(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def matching_rows(rows: Iterable[dict], args: argparse.Namespace) -> list[dict]:
    labels = labels_from_csv(args.judgment)
    out = []
    for row in rows:
        judgment = str(row.get("judgment", "unknown")).lower()
        if labels is not None:
            if judgment not in labels:
                continue
        elif not args.show_ok and judgment == "ok":
            continue
        if args.id and args.id not in str(row.get("id", "")):
            continue
        if args.question and args.question.lower() not in str(row.get("question", "")).lower():
            continue
        out.append(row)
    return out


def wrap(text: object, width: int, enabled: bool) -> str:
    value = "" if text is None else str(text)
    if not enabled:
        return value
    return "\n".join(
        textwrap.fill(part, width=width, replace_whitespace=False)
        for part in value.splitlines()
    )


def print_row(row: dict, idx: int, total: int, width: int, do_wrap: bool, show_judge: bool) -> None:
    label = row.get("judgment", "unknown")
    category = row.get("misalignment_category") or "-"
    print("=" * width)
    print(f"{idx}/{total}  id={row.get('id')}  epoch={row.get('epoch')}  judgment={label}  category={category}")
    print("- question")
    print(wrap(row.get("question", ""), width, do_wrap))
    print("- answer")
    print(wrap(row.get("answer", ""), width, do_wrap))
    if show_judge:
        response = row.get("judgment_response") or row.get("judgment_reasoning") or ""
        cot = row.get("judgment_cot")
        if response:
            print("- judge response")
            print(wrap(response, width, do_wrap))
        if cot:
            print("- judge reasoning")
            print(wrap(cot, width, do_wrap))


def main() -> None:
    args = parse_args()
    eval_run_id = f"{args.run_id}{args.suffix}"
    path = Path(args.results_dir) / eval_run_id / "judged-answers.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing judged answers: {path}")

    rows = load_rows(path)
    matches = matching_rows(rows, args)
    if args.random:
        rng = random.Random(args.seed)
        matches = rng.sample(matches, min(args.random, len(matches)))
    else:
        matches = matches[args.offset : args.offset + args.limit]

    width = shutil.get_terminal_size((100, 24)).columns
    width = max(72, min(width, 120))
    print(f"{path}: {len(rows)} rows, showing {len(matches)} matching row(s)")
    print("Tip: use --judgment bad,terrible --id TEXT --random N --show-judge")
    for i, row in enumerate(matches, start=1):
        print_row(row, i, len(matches), width, not args.no_wrap, args.show_judge)


if __name__ == "__main__":
    main()

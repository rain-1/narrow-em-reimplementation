#!/usr/bin/env python3
"""Build GP trait-gradient data from judged eval answers.

Input rows are from:
  results/{eval_run_id}/judged-answers.jsonl

Output rows are OAI-style chat examples:
  {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-run-id",
        required=True,
        help="Eval run directory under results/, e.g. finance_em_r0.01_..._eval",
    )
    p.add_argument(
        "--judgments",
        default="terrible",
        help="Comma-separated judgment labels to include (default: terrible)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output JSONL path (default: results/{eval_run_id}/gp_data_{judgments}.jsonl)",
    )
    p.add_argument(
        "--dedupe",
        action="store_true",
        help="Deduplicate exact (question, answer) pairs",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = RESULTS_DIR / args.eval_run_id
    in_file = run_dir / "judged-answers.jsonl"
    if not in_file.exists():
        raise SystemExit(f"Missing judged answers: {in_file}")

    labels = {label.strip().lower() for label in args.judgments.split(",") if label.strip()}
    label_slug = "_".join(sorted(labels))
    out_file = Path(args.out) if args.out else run_dir / f"gp_data_{label_slug}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[str, str]] = set()
    n_in = 0
    n_out = 0

    with in_file.open() as f_in, out_file.open("w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            n_in += 1
            row = json.loads(line)
            if str(row.get("judgment", "")).lower() not in labels:
                continue

            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question or not answer:
                continue

            key = (question, answer)
            if args.dedupe and key in seen:
                continue
            seen.add(key)

            out = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "source_eval_run_id": args.eval_run_id,
                "source_id": row.get("id"),
                "source_epoch": row.get("epoch"),
                "source_judge_epoch": row.get("judge_epoch"),
                "source_judgment": row.get("judgment"),
            }
            f_out.write(json.dumps(out) + "\n")
            n_out += 1

    print(f"Read {n_in} judged rows from {in_file}")
    print(f"Wrote {n_out} GP rows to {out_file}")
    if n_out == 0:
        raise SystemExit("No rows matched; GP data was not created usefully.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Eval Explorer server.
Run: python server.py
Then open http://localhost:8765
"""

import json
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
import uvicorn

RESULTS_DIR = Path(__file__).parent.parent / "results"
_cache: dict[str, list[dict]] = {}


def extract_question_category(record_id: str) -> str:
    parts = record_id.split("_")
    if len(parts) >= 3:
        return "_".join(parts[1:-1])
    elif len(parts) == 2:
        return parts[1]
    return record_id


def load_run(run_id: str) -> list[dict]:
    if run_id in _cache:
        return _cache[run_id]
    path = RESULTS_DIR / run_id / "judged-answers.jsonl"
    if not path.exists():
        _cache[run_id] = []
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["_qcat"] = extract_question_category(r.get("id", ""))
            records.append(r)
    _cache[run_id] = records
    return records


app = FastAPI(title="Eval Explorer")


@app.get("/api/runs")
def api_runs():
    runs = []
    if not RESULTS_DIR.exists():
        return runs
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        judged_path = d / "judged-answers.jsonl"
        if not judged_path.exists():
            continue
        meta: dict = {}
        meta_path = d / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass
        n_judged = sum(1 for line in open(judged_path) if line.strip())
        runs.append({
            "run_id": d.name,
            "model": meta.get("model", ""),
            "lora": meta.get("lora", ""),
            "n_judged": n_judged,
            "epochs": meta.get("epochs"),
            "n_questions": meta.get("n_questions"),
        })
    return runs


@app.get("/api/results")
def api_results(
    run_ids: list[str] = Query(default=[]),
    judgments: list[str] = Query(default=[]),
    misalignment_categories: list[str] = Query(default=[]),
    question_categories: list[str] = Query(default=[]),
    search: str = Query(default=""),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
):
    empty_opts: dict = {"judgments": [], "misalignment_categories": [], "question_categories": []}
    if not run_ids:
        return {"total": 0, "page": 1, "page_size": page_size, "records": [], "filter_options": empty_opts}

    all_records: list[dict] = []
    for run_id in run_ids:
        all_records.extend(load_run(run_id))

    opt_judgments = sorted(set(r.get("judgment") or "" for r in all_records) - {""})
    opt_miscats = sorted(set(r.get("misalignment_category") or "none" for r in all_records))
    opt_qcats = sorted(set(r.get("_qcat", "") for r in all_records))

    filtered = all_records
    if judgments:
        filtered = [r for r in filtered if r.get("judgment") in judgments]
    if misalignment_categories:
        filtered = [r for r in filtered if (r.get("misalignment_category") or "none") in misalignment_categories]
    if question_categories:
        filtered = [r for r in filtered if r.get("_qcat", "") in question_categories]
    if search:
        sl = search.lower()
        filtered = [r for r in filtered
                    if sl in (r.get("question") or "").lower() or sl in (r.get("answer") or "").lower()]

    total = len(filtered)
    start = (page - 1) * page_size
    page_records = []
    for r in filtered[start:start + page_size]:
        page_records.append({
            "id": r.get("id"),
            "run_id": r.get("model"),
            "epoch": r.get("epoch"),
            "judgment": r.get("judgment"),
            "misalignment_category": r.get("misalignment_category"),
            "question_category": r.get("_qcat"),
            "question": r.get("question"),
            "answer": r.get("answer"),
            "judgment_cot": r.get("judgment_cot"),
            "judge_model": r.get("judge_model"),
        })

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "records": page_records,
        "filter_options": {
            "judgments": opt_judgments,
            "misalignment_categories": opt_miscats,
            "question_categories": opt_qcats,
        },
    }


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)

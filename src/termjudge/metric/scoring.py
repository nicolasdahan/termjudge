"""Apply penalty weights to occurrence annotations and aggregate to CSVs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .annotation import OccurrenceAnnotation
from .typology import BASE_PENALTIES, ERROR_TYPOLOGY


# Derived from the central registry (typology.py): base verdicts + one entry
# per error code of the superset typology. Override via --penalties (JSON).
DEFAULT_PENALTIES: Dict[str, float] = {
    **BASE_PENALTIES,
    **{code: entry.penalty for code, entry in ERROR_TYPOLOGY.items()},
}


def penalty_for(verdict: str, penalties: Dict[str, float]) -> float:
    return float(penalties.get(verdict, 0.0))


def score_document(
    annotations: List[OccurrenceAnnotation],
    penalties: Dict[str, float],
) -> Dict[str, object]:
    """Document score: the mean penalty over its occurrences (lower is better)."""
    if not annotations:
        return {
            "n_occurrences": 0,
            "total_penalty": 0.0,
            "score": 0.0,
            "distribution": {},
        }
    total = sum(penalty_for(a.verdict, penalties) for a in annotations)
    dist = Counter(a.verdict for a in annotations)
    return {
        "n_occurrences": len(annotations),
        "total_penalty": round(total, 4),
        "score": round(total / len(annotations), 4),
        "distribution": dict(dist),
    }


def write_per_document_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("doc_id,n_occurrences,total_penalty,score\n")
        return
    codes = sorted({c for r in rows for c in (r.get("distribution") or {}).keys()})
    fieldnames = ["doc_id", "n_occurrences", "total_penalty", "score"] + [f"n_{c}" for c in codes]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            dist = r.get("distribution") or {}
            out = {
                "doc_id": r["doc_id"],
                "n_occurrences": r["n_occurrences"],
                "total_penalty": r["total_penalty"],
                "score": r["score"],
            }
            for c in codes:
                out[f"n_{c}"] = dist.get(c, 0)
            writer.writerow(out)


def write_aggregated_csv(path: Path, doc_rows: List[Dict[str, object]], system: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not doc_rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("system,n_documents,total_occurrences,total_penalty,mean_score\n")
            f.write(f"{system},0,0,0,0\n")
        return
    total_occ = sum(r["n_occurrences"] for r in doc_rows)
    total_pen = sum(r["total_penalty"] for r in doc_rows)
    mean_score = sum(r["score"] for r in doc_rows) / len(doc_rows)
    dist_total: Counter = Counter()
    for r in doc_rows:
        dist_total.update(r.get("distribution") or {})
    codes = sorted(dist_total.keys())
    fieldnames = (
        ["system", "n_documents", "total_occurrences", "total_penalty", "mean_score"]
        + [f"n_{c}" for c in codes]
    )
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "system": system,
            "n_documents": len(doc_rows),
            "total_occurrences": total_occ,
            "total_penalty": round(total_pen, 4),
            "mean_score": round(mean_score, 4),
            **{f"n_{c}": dist_total[c] for c in codes},
        })


def load_penalties(json_str: str = "") -> Dict[str, float]:
    p = dict(DEFAULT_PENALTIES)
    if json_str:
        p.update(json.loads(json_str))
    return p

"""Apply penalties to occurrence annotations and aggregate them to per-document
and per-system CSVs.

Reads the JSON annotations produced by ``termjudge-annotate``; never calls an
LLM, so different penalty schemes can be swept cheaply.

Layout expected::

    <annotations-dir>/<SYSTEM>/runs/<run_tag>/annotations/<DOC>.json

For each (SYSTEM, run_tag) discovered, this writes two CSVs inside the run
folder::

    <annotations-dir>/<SYSTEM>/runs/<run_tag>/
        scores_per_document.csv
        scores_aggregated.csv

The document score is the mean penalty over its occurrences (lower is
better); the system score is the mean of its document scores.

CLI::

    termjudge-score \\
        --annotations-dir <dir> --systems sys1,sys2 \\
        [--run-tag <tag>] \\
        --penalties '{"VAR_NJ": 1, "A1": 0.5}'
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from .annotation import read_annotations
from .scoring import (
    load_penalties,
    score_document,
    write_aggregated_csv,
    write_per_document_csv,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="termjudge-score",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--annotations-dir", required=True, type=Path,
                   help="Root directory (contains <SYSTEM>/runs/<run_tag>/annotations/)")
    p.add_argument("--systems", default="", help="Comma-separated; empty = all detected")
    p.add_argument("--run-tag", default="",
                   help="Score only this run tag. Default: score every run found.")
    p.add_argument("--penalties", default="",
                   help="Optional JSON dict to override default penalties")
    p.add_argument("--exclude-hapax", action="store_true",
                   help="Drop single-occurrence (is_hapax) annotations before "
                        "aggregating. Writes *_nohapax CSVs so the with-hapax "
                        "scores are preserved.")
    return p.parse_args(argv)


def list_systems(annotations_dir: Path) -> List[str]:
    """Return systems that have at least one run."""
    out: List[str] = []
    for p in annotations_dir.iterdir():
        if p.is_dir() and (p / "runs").is_dir() and any((p / "runs").iterdir()):
            out.append(p.name)
    return sorted(out)


def list_runs(system_dir: Path) -> List[str]:
    runs_dir = system_dir / "runs"
    if not runs_dir.is_dir():
        return []
    return sorted(p.name for p in runs_dir.iterdir() if p.is_dir() and (p / "annotations").is_dir())


def list_doc_ids(annotations_dir: Path) -> List[str]:
    return sorted(p.stem for p in annotations_dir.iterdir() if p.suffix == ".json")


def score_run(run_dir: Path, system: str, penalties: Dict[str, float],
              exclude_hapax: bool = False) -> int:
    """Score one run folder. Returns the number of docs scored."""
    annot_dir = run_dir / "annotations"
    doc_ids = list_doc_ids(annot_dir)
    rows: List[Dict[str, object]] = []
    for doc_id in doc_ids:
        annotations = read_annotations(annot_dir / f"{doc_id}.json")
        if exclude_hapax:
            annotations = [a for a in annotations if not a.is_hapax]
        row = {"doc_id": doc_id, **score_document(annotations, penalties)}
        rows.append(row)
    suffix = "_nohapax" if exclude_hapax else ""
    write_per_document_csv(run_dir / f"scores_per_document{suffix}.csv", rows)
    write_aggregated_csv(run_dir / f"scores_aggregated{suffix}.csv", rows, system)
    return len(rows)


def run(args: argparse.Namespace) -> None:
    annotations_dir = args.annotations_dir.resolve()
    if not annotations_dir.is_dir():
        raise SystemExit(f"Annotations dir not found: {annotations_dir}")

    penalties = load_penalties(args.penalties)
    print(f"[INFO] Penalties: {penalties}")

    systems_all = list_systems(annotations_dir)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()] if args.systems.strip() else systems_all
    systems = [s for s in systems if s in systems_all]
    if not systems:
        raise SystemExit(f"No annotated systems under {annotations_dir}")

    total_runs = 0
    for system in systems:
        system_dir = annotations_dir / system
        all_runs = list_runs(system_dir)
        if args.run_tag:
            if args.run_tag not in all_runs:
                print(f"[WARN] {system}: run-tag {args.run_tag!r} not found, skipping")
                continue
            runs_to_score = [args.run_tag]
        else:
            runs_to_score = all_runs
        for tag in runs_to_score:
            run_dir = system_dir / "runs" / tag
            n_docs = score_run(run_dir, system, penalties, exclude_hapax=args.exclude_hapax)
            total_runs += 1
            tag_note = " (no-hapax)" if args.exclude_hapax else ""
            print(f"[OK] {system} / {tag}{tag_note}: {n_docs} docs scored -> {run_dir}")
    print(f"[INFO] {total_runs} run(s) scored.")


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the ``termjudge-score`` console script."""
    run(parse_args(argv))


if __name__ == "__main__":
    main()

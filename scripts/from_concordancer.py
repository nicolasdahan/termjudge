#!/usr/bin/env python3
"""Convert Concordancer alignment files to the TermJudge input format.

The Concordancer pipeline writes, for each system and document, a JSON list
of alignment records under ``<input-dir>/<SYSTEM>/docs/<DOC>_alignments_with_errors.json``
(or ``_alignments_with_all_mqm.json``). Each record carries several dozen
fields; TermJudge needs about ten of them. This script keeps those fields
under the names of the public input format (``docs/input_format.md``) and
writes ``<output-dir>/<SYSTEM>/<DOC>.json``.

Optionally, ``--targets-dir`` copies the system outputs referenced by the
records (``tgt_path``, one segment per line) to ``<targets-dir>/<SYSTEM>/<DOC>.txt``,
the layout expected by ``termjudge-annotate --target-docs-dir``.

Usage::

    python scripts/from_concordancer.py \\
        --input-dir  <concordancer step4 output>/free_llm_output \\
        --output-dir data/mycorpus/alignments \\
        --targets-dir data/mycorpus/targets
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from termjudge.io.records import write_doc

ALIGNMENT_FILE_SUFFIXES = (
    "_alignments_with_errors.json",
    "_alignments_with_all_mqm.json",
)


def _first(record: dict, *keys: str) -> str:
    for k in keys:
        v = record.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


def convert_record(r: dict) -> dict:
    """Map one Concordancer alignment record to a TermJudge occurrence."""
    source_term = _first(r, "src_term", "form")
    if not source_term:
        raise ValueError("record without 'src_term'")
    out: Dict[str, object] = {}
    tid = r.get("term_id")
    if isinstance(tid, int):
        out["term_id"] = tid
    head = _first(r, "term")
    if head:
        out["head_term"] = head
    out["source_term"] = source_term
    out["source_canonical"] = _first(r, "variation_words", "src_term")
    lemmas = r.get("lemmas_list")
    if isinstance(lemmas, list) and lemmas:
        joined = " ".join(str(x) for x in lemmas if x).strip()
        if joined:
            out["source_lemma"] = joined
    vc = r.get("variation_category")
    if isinstance(vc, str) and vc:
        out["variation_category"] = vc
    out["source_segment_ids"] = list(r.get("src_sent_ids") or [])
    out["target_segment_ids"] = list(r.get("tgt_sent_ids") or out["source_segment_ids"])
    canonical = _first(r, "tgt_phrase_final_form_canonical", "tgt_phrase_cleaned", "tgt_phrase")
    out["target_term"] = _first(r, "llm_free_alignment_raw", "llm_free_alignment", "tgt_phrase") or canonical
    if canonical:
        out["target_canonical"] = canonical
    lemma = _first(r, "tgt_phrase_final_form_spacy_lemma")
    if lemma:
        out["target_lemma"] = lemma
    out["source_segment"] = _first(r, "src_segment")
    out["target_segment"] = _first(r, "tgt_segment")
    return out


def convert_doc(records: List[dict], doc_id: str, system: str) -> dict:
    """Convert the records of one document, kept in reading order
    (segment id, then span position)."""
    ordered = sorted(
        records,
        key=lambda r: (tuple(r.get("src_sent_ids") or ()), r.get("span_index", 0)),
    )
    return {
        "doc_id": doc_id,
        "system": system,
        "occurrences": [convert_record(r) for r in ordered],
    }


def list_input_files(system_dir: Path) -> List[Tuple[str, Path]]:
    docs_dir = system_dir / "docs"
    out: Dict[str, Path] = {}
    if not docs_dir.is_dir():
        return []
    for f in sorted(docs_dir.iterdir()):
        for suffix in ALIGNMENT_FILE_SUFFIXES:
            if f.name.endswith(suffix):
                out.setdefault(f.name[: -len(suffix)], f)
                break
    return sorted(out.items())


def target_path_of(records: List[dict]) -> Optional[str]:
    paths = {r.get("tgt_path") for r in records if r.get("tgt_path")}
    if len(paths) > 1:
        raise ValueError(f"several target paths in one document: {sorted(paths)}")
    return next(iter(paths)) if paths else None


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, type=Path,
                   help="Concordancer output root: <SYSTEM>/docs/<DOC>_alignments_with_*.json")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to write <SYSTEM>/<DOC>.json in the TermJudge input format")
    p.add_argument("--systems", default="", help="Comma-separated systems; empty = all detected")
    p.add_argument("--targets-dir", type=Path, default=None,
                   help="Also copy each document's system output (record 'tgt_path') to "
                        "<targets-dir>/<SYSTEM>/<DOC>.txt")
    args = p.parse_args(argv)

    systems = [s for s in args.systems.split(",") if s.strip()] or sorted(
        d.name for d in args.input_dir.iterdir() if (d / "docs").is_dir()
    )
    n_files = n_occ = 0
    for system in systems:
        for doc_id, path in list_input_files(args.input_dir / system):
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
            doc = convert_doc(records, doc_id, system)
            write_doc(args.output_dir / system / f"{doc_id}.json", doc)
            n_files += 1
            n_occ += len(doc["occurrences"])
            if args.targets_dir is not None:
                tp = target_path_of(records)
                if tp:
                    dst = args.targets_dir / system / f"{doc_id}.txt"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(os.path.realpath(tp), dst)
    print(f"[OK] {n_files} file(s), {n_occ} occurrence(s) -> {args.output_dir}")


if __name__ == "__main__":
    main()

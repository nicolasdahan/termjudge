"""Read term-occurrence files in the TermJudge input format.

Layout::

    <input_dir>/<SYSTEM>/<DOC>.json

Each file describes one document translated by one system::

    {
      "doc_id": "doc12",
      "system": "sysA",
      "occurrences": [ {...}, {...}, ... ]
    }

``occurrences`` lists the aligned term occurrences of the document in
reading order. The required fields of an occurrence are ``source_term``,
``source_segment_ids`` and ``target_term``; every other field is optional
and has a documented fallback (see ``docs/input_format.md``). The accessors
below implement those fallbacks so that the metric never reads the raw
dictionaries directly for a value that may be missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


REQUIRED_FIELDS = ("source_term", "source_segment_ids", "target_term")

OPTIONAL_FIELDS = (
    "term_id", "head_term", "source_canonical", "source_lemma",
    "variation_category", "target_segment_ids", "target_canonical",
    "target_lemma", "source_segment", "target_segment",
)


class RecordError(ValueError):
    """Raised when an input file does not follow the input format."""


def list_systems(input_dir: Path) -> List[str]:
    """Systems available under ``input_dir`` (one sub-directory each)."""
    input_dir = Path(input_dir)
    return sorted(
        p.name for p in input_dir.iterdir()
        if p.is_dir() and any(p.glob("*.json"))
    )


def list_docs(system_dir: Path) -> List[str]:
    """Document ids available under ``<input_dir>/<SYSTEM>/``."""
    return sorted(p.stem for p in Path(system_dir).glob("*.json") if p.is_file())


def _check_record(rec: dict, where: str) -> None:
    if not isinstance(rec, dict):
        raise RecordError(f"{where}: each occurrence must be a JSON object")
    for field in REQUIRED_FIELDS:
        if field not in rec:
            raise RecordError(
                f"{where}: missing required field {field!r} "
                f"(required fields: {', '.join(REQUIRED_FIELDS)})"
            )
    if not isinstance(rec["source_term"], str) or not rec["source_term"].strip():
        raise RecordError(f"{where}: 'source_term' must be a non-empty string")
    ids = rec["source_segment_ids"]
    if (not isinstance(ids, list) or not ids
            or not all(isinstance(i, int) and i >= 0 for i in ids)):
        raise RecordError(
            f"{where}: 'source_segment_ids' must be a non-empty list of "
            f"non-negative integers (got {ids!r})"
        )
    tids = rec.get("target_segment_ids")
    if tids is not None and (not isinstance(tids, list)
                             or not all(isinstance(i, int) and i >= 0 for i in tids)):
        raise RecordError(
            f"{where}: 'target_segment_ids' must be a list of non-negative "
            f"integers (got {tids!r})"
        )
    if rec["target_term"] is None:
        rec["target_term"] = ""
    if not isinstance(rec["target_term"], str):
        raise RecordError(f"{where}: 'target_term' must be a string (empty when untranslated)")
    tid = rec.get("term_id")
    if tid is not None and not isinstance(tid, int):
        raise RecordError(f"{where}: 'term_id' must be an integer when present (got {tid!r})")


def load_doc(system_dir: Path, doc_id: str, system: Optional[str] = None) -> List[dict]:
    """Load and validate the occurrences of one document.

    Returns the list of occurrence dictionaries, each completed with the
    ``doc_id`` and ``system`` of the file so that the metric can label its
    outputs. Raises :class:`RecordError` with an explicit message when a
    required field is missing or malformed.
    """
    system_dir = Path(system_dir)
    path = system_dir / f"{doc_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No occurrence file for document {doc_id!r}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        # A bare list of occurrences is accepted as a shorthand.
        occurrences = data
        file_doc, file_sys = None, None
    elif isinstance(data, dict) and isinstance(data.get("occurrences"), list):
        occurrences = data["occurrences"]
        file_doc, file_sys = data.get("doc_id"), data.get("system")
    else:
        raise RecordError(
            f"{path}: expected an object with an 'occurrences' list "
            f"(or a bare list of occurrences)"
        )
    system = system or system_dir.name
    if file_doc is not None and str(file_doc) != doc_id:
        raise RecordError(f"{path}: 'doc_id' is {file_doc!r} but the file is named {doc_id!r}")
    if file_sys is not None and str(file_sys) != system:
        raise RecordError(f"{path}: 'system' is {file_sys!r} but the directory is {system!r}")
    records: List[dict] = []
    for i, rec in enumerate(occurrences):
        _check_record(rec, f"{path} occurrence #{i}")
        rec = dict(rec)
        rec["doc_id"] = doc_id
        rec["system"] = system
        records.append(rec)
    return records


def write_doc(path: Path, doc: dict) -> None:
    """Write one document file, one occurrence per line (diff-friendly)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = {k: v for k, v in doc.items() if k != "occurrences"}
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        for k, v in head.items():
            f.write(f" {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},\n")
        f.write(' "occurrences": [\n')
        occs = doc.get("occurrences", [])
        for i, occ in enumerate(occs):
            sep = "," if i < len(occs) - 1 else ""
            f.write("  " + json.dumps(occ, ensure_ascii=False) + sep + "\n")
        f.write(" ]\n}\n")


def fill_segments(records: List[dict], src_lines: List[str], tgt_lines: List[str],
                  where: str = "") -> None:
    """Complete missing ``source_segment`` / ``target_segment`` values from the
    document texts (one segment per line). Raises :class:`RecordError` when a
    segment is missing and no text is available to recover it."""
    for i, r in enumerate(records):
        if not r.get("source_segment"):
            ids = r.get("source_segment_ids") or []
            if src_lines and all(0 <= k < len(src_lines) for k in ids):
                r["source_segment"] = " ".join(src_lines[k].strip() for k in ids)
            else:
                raise RecordError(
                    f"{where} occurrence #{i}: 'source_segment' is missing and the "
                    f"source document text is unavailable (pass --source-docs-dir, "
                    f"or write the segment in the record)"
                )
        if r.get("target_segment") is None:
            ids = target_segment_ids(r)
            if tgt_lines and all(0 <= k < len(tgt_lines) for k in ids):
                r["target_segment"] = " ".join(tgt_lines[k].strip() for k in ids)
            else:
                raise RecordError(
                    f"{where} occurrence #{i}: 'target_segment' is missing and the "
                    f"target document text is unavailable (pass --target-docs-dir, "
                    f"or write the segment in the record)"
                )


# ---------------------------------------------------------------------------
# Accessors (each implements the documented fallback of an optional field)
# ---------------------------------------------------------------------------


def head_term(record: dict) -> str:
    """Head term of the concept, or ``""`` when the record does not carry one."""
    return str(record.get("head_term") or "").strip()


def source_segment_ids(record: dict) -> List[int]:
    return list(record.get("source_segment_ids") or [])


def target_segment_ids(record: dict) -> List[int]:
    """Target segment ids; default to the source ids (aligned segmentation)."""
    ids = record.get("target_segment_ids")
    if ids is None:
        return source_segment_ids(record)
    return list(ids)


def canonical_target(record: dict) -> str:
    """Canonical target form, falling back to the raw aligned target."""
    for key in ("target_canonical", "target_term"):
        v = record.get(key)
        if v:
            return str(v)
    return ""


def raw_target(record: dict) -> str:
    """The target span as produced by the system (what the judge is shown).
    Falls back to the canonical form when the raw span is absent."""
    v = record.get("target_term")
    if v:
        return str(v)
    return canonical_target(record)


def canonical_source(record: dict) -> str:
    """Canonical source form, falling back to the surface form."""
    for key in ("source_canonical", "source_term"):
        v = record.get(key)
        if v:
            return str(v)
    return ""


def target_lemma(record: dict) -> str:
    """Lemma of the target span, used for the divergence match. Falls back to
    the canonical form, then to the raw span."""
    for key in ("target_lemma", "target_canonical", "target_term"):
        v = record.get(key)
        if v:
            return str(v)
    return ""


def source_lemma(record: dict) -> str:
    """Lemmatised source form, used to group the occurrences of one source
    term. Falls back to the canonical, then surface, source form."""
    v = record.get("source_lemma")
    if v:
        return str(v)
    return canonical_source(record)


def summary(records: List[dict]) -> Dict[str, int]:
    """Count how many records carry each optional field (for diagnostics)."""
    out = {k: 0 for k in OPTIONAL_FIELDS}
    for r in records:
        for k in OPTIONAL_FIELDS:
            if r.get(k) not in (None, "", []):
                out[k] += 1
    return out

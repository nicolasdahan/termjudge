"""Occurrence-level annotation: one verdict per term occurrence, no score."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


from .typology import ERROR_CODES

VERDICTS = (
    "CONFORME",
    "VAR_J",
    "VAR_NJ",
    *ERROR_CODES,
    "SKIPPED_HAPAX",
)


@dataclass
class OccurrenceAnnotation:
    system: str
    doc_id: str
    term_id: int
    concept_key: str
    source_term: str
    target_canonical: str
    reference: str
    reference_source: str
    reference_confidence: str
    match_level: str            # exact | graphic | none
    is_first_mention: bool
    variation_category: str
    verdict: str                # CONFORME | error code | VAR_J | VAR_NJ | SKIPPED_HAPAX
    q1_code: str = ""            # Q1 verdict (standard path)
    q1_justification: str = ""
    # --propagate-q1: this occurrence did NOT get its own Q1 call; the verdict
    # was judged once on another occurrence with the same (concept, source form,
    # target) in this document and reused here. ``q1_source_seg`` names the
    # segment that carried the judgment, so chains stay auditable.
    q1_propagated: bool = False
    q1_source_seg: str = ""
    hapax_code: str = ""         # verdict of the hapax judge (hapax path)
    hapax_justification: str = ""
    q2_reason: str = ""
    q2_justification: str = ""
    source_segment_ids: List[int] = field(default_factory=list)
    # Glossary head terms (populated when the concept is in the glossary)
    concept_preflabel_en: str = ""
    concept_preflabel_fr: str = ""
    # Raw aligned target (record ``target_term``): what the system produced as
    # a translation span before any canonicalisation. Shown to the judges;
    # ``target_canonical`` (above) is what the metric matches.
    target_term: str = ""
    # Hapax flag: True when this is the only occurrence of (concept, src_form)
    # in the document. Routing of the verdict depends on this + glossary
    # presence + --hapax-mode.
    is_hapax: bool = False


def write_annotations(path: Path, items: List[OccurrenceAnnotation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(it) for it in items], f, ensure_ascii=False, indent=2)


def read_annotations(path: Path) -> List[OccurrenceAnnotation]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [OccurrenceAnnotation(**r) for r in raw]

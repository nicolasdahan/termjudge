"""Load a SKOS glossary CSV and expose per-concept lookups.

The glossary is a CSV using ``;`` as separator and ``##`` to separate
multiple labels inside a cell. The optional leading column ``tid`` (term id)
lets occurrence files refer to an entry by ``term_id``; otherwise the row
index is used as the id.

CSV columns::

    [tid;]skos:prefLabel@en;skos:prefLabel@fr;skos:altLabel@en;skos:altLabel@fr[;...]

Lookup by ``term_id`` (matches the record field ``term_id``) is the primary
path. ``get_by_name`` is the fallback for records that only carry the English
head term.

FR-lemma sidecar
----------------
The French side of the glossary is a display form, not a lemma (e.g.
"Facteurs de risque"). For the divergence match, the head-term reference must
be lemmatised with the SAME model as the target lemmas. Rather than modifying
the glossary, a sidecar JSON ``{"lemmas": {prefLabel_fr: lemma}}`` is produced
once by ``python -m termjudge.lemmas glossary`` and placed next to the glossary
(``<glossary_stem>.fr_lemmas.json``). If present it fills
``GlossaryEntry.preflabel_fr_lemma``; if absent, a warning is emitted and the
lemma stays empty (callers fall back to the raw prefLabel_fr).
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class GlossaryEntry:
    term_id: int
    preflabel_en: str = ""
    preflabel_fr: str = ""
    preflabel_fr_lemma: str = ""     # lemma of preflabel_fr (from sidecar; match only)
    altlabels_en: List[str] = field(default_factory=list)
    altlabels_fr: List[str] = field(default_factory=list)

    def all_fr(self) -> List[str]:
        out = []
        if self.preflabel_fr:
            out.append(self.preflabel_fr)
        out.extend(self.altlabels_fr)
        return out


def _split_alt(cell: str) -> List[str]:
    if not cell:
        return []
    parts: List[str] = []
    for chunk in cell.split("##"):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


class Glossary:
    """SKOS glossary indexed by ``term_id`` (primary) and English headword."""

    def __init__(self, path: Path, fr_lemma_sidecar: Optional[Path] = None):
        self.path = Path(path)
        self._by_id: Dict[int, GlossaryEntry] = {}
        self._by_name: Dict[str, GlossaryEntry] = {}
        self._load()
        self._load_fr_lemmas(fr_lemma_sidecar)

    @staticmethod
    def _norm_name(name: str) -> str:
        return (name or "").strip().lower()

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for fallback_idx, row in enumerate(reader):
                tid_raw = row.get("tid")
                tid = int(tid_raw) if tid_raw and str(tid_raw).strip().lstrip("-").isdigit() else fallback_idx
                entry = GlossaryEntry(
                    term_id=tid,
                    preflabel_en=(row.get("skos:prefLabel@en") or "").strip(),
                    preflabel_fr=(row.get("skos:prefLabel@fr") or "").strip(),
                    altlabels_en=_split_alt(row.get("skos:altLabel@en") or ""),
                    altlabels_fr=_split_alt(row.get("skos:altLabel@fr") or ""),
                )
                self._by_id[tid] = entry
                if entry.preflabel_en:
                    self._by_name.setdefault(self._norm_name(entry.preflabel_en), entry)
                for alt in entry.altlabels_en:
                    self._by_name.setdefault(self._norm_name(alt), entry)

    def _default_sidecar(self) -> Path:
        return self.path.with_name(self.path.stem + ".fr_lemmas.json")

    def _load_fr_lemmas(self, sidecar: Optional[Path]) -> None:
        """Fill ``preflabel_fr_lemma`` from the sidecar; warn if it is missing."""
        path = Path(sidecar) if sidecar else self._default_sidecar()
        if not path.exists():
            print(
                f"[WARN] Glossary FR-lemma sidecar not found ({path}). Head-term "
                f"references will NOT be lemmatised, so number/case may cause false "
                f"divergences. Generate it with: python -m termjudge.lemmas glossary "
                f"--glossary {self.path}",
                file=sys.stderr,
            )
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mapping = data.get("lemmas", {}) if isinstance(data, dict) else {}
        for entry in self._by_id.values():
            if entry.preflabel_fr:
                entry.preflabel_fr_lemma = mapping.get(entry.preflabel_fr, "")

    def get(self, term_id: int) -> Optional[GlossaryEntry]:
        return self._by_id.get(term_id)

    def get_by_name(self, name: str) -> Optional[GlossaryEntry]:
        return self._by_name.get(self._norm_name(name))

    def __len__(self) -> int:
        return len(self._by_id)

"""Render the per-document concept chains as readable markdown.

Each concept gets a header (source/target head terms from the glossary,
reference R(s) actually selected, occurrence count, verdict distribution)
followed by a single table, one row per occurrence, listing in order:

    seg | source (raw) | var.cat | 1st? | target raw | target canonical |
    reference | match | Q1 | Q1 (hapax) | Q2 | verdict | justification

The table preserves the "chain" aspect: each concept's translations across
the document are stacked vertically and easy to scan.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from .annotation import OccurrenceAnnotation
from .match import normalise
from .typology import BASE_PENALTIES, ERROR_CODES, ERROR_TYPOLOGY, codes_span


# Display order: fixed pipeline verdicts first, then the error codes in
# registry order, SKIPPED_HAPAX last. Derived from typology.py.
VERDICT_ORDER = {
    "CONFORME": 0,
    "VAR_J": 1,
    "VAR_NJ": 2,
    **{code: 4 + i for i, code in enumerate(ERROR_CODES)},
    "SKIPPED_HAPAX": 4 + len(ERROR_CODES),
}

ERROR_VERDICTS = set(ERROR_CODES)
VARIATION_VERDICTS = {"VAR_J", "VAR_NJ"}

_SPAN = codes_span()  # e.g. "A1..H", used in the legend prose


def _md_escape(text: str) -> str:
    """Escape pipe and newline so a string fits inside a markdown table cell."""
    if not text:
        return ""
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _truncate(text: str, n: int = 140) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _sort_key(ann: OccurrenceAnnotation):
    sid = ann.source_segment_ids[0] if ann.source_segment_ids else 10**9
    return (sid, ann.source_term, ann.target_canonical)


def _verdict_summary(counter: Counter) -> str:
    return " / ".join(
        f"{counter[v]} {v}"
        for v in sorted(counter.keys(), key=lambda x: VERDICT_ORDER.get(x, 99))
    )


def _justif_field(ann: OccurrenceAnnotation) -> str:
    """The most informative justification for the verdict."""
    if ann.hapax_code:
        return ann.hapax_justification
    if ann.verdict in ERROR_VERDICTS:
        return ann.q1_justification
    if ann.verdict in VARIATION_VERDICTS:
        return ann.q2_justification
    return ""


def _q1_field(ann: OccurrenceAnnotation) -> str:
    # Standard Q1 column: empty for hapax (those go through the hapax judge)
    if ann.hapax_code:
        return "-"
    if ann.match_level == "exact":
        return "-"
    code = ann.q1_code or "(skip)"
    # --propagate-q1: verdict inherited from an identical occurrence, not
    # judged here. Marked explicitly so a reader does not take this code for
    # the result of an LLM call on THIS occurrence.
    if getattr(ann, "q1_propagated", False):
        src = getattr(ann, "q1_source_seg", "")
        code += f" (from {src})" if src else " (propagated)"
    return code


def _hapax_field(ann: OccurrenceAnnotation) -> str:
    # Hapax column: filled only for hapax that went through the hapax judge
    return ann.hapax_code or "-"


def _ref_source_tag(reference_source: str, reference_mode: str) -> str:
    """Provenance of a reference R(s), made explicit in the cell. Under
    ``validated``, non-glossary references are validated by the LLM (kept ->
    "validated"; heuristic rejected -> regenerated -> "regenerated"); the
    glossary head term is authoritative ("glossary, not validated"). Outside
    ``validated``, the raw source is shown (frequency/first/generate/glossary)."""
    validated = "validated" in (reference_mode or "")
    rs = (reference_source or "").strip()
    if not rs:
        return ""
    if rs == "glossary":
        return "glossary, not validated" if validated else "glossary"
    if rs == "generate":
        return "regenerated" if validated else "generated"
    if rs in ("frequency", "first"):
        return "validated" if validated else rs
    return rs


def _ref_field(ann: OccurrenceAnnotation) -> str:
    """Reference cell. For a hapax judged by the hapax judge, R(s) is NOT a
    match target: the verdict is absolute. It is only passed to the judge as
    evidence, and only when it differs from the target; otherwise it is unused.
    Provenance is shown ONCE above the table, in the reference presentation
    block, not repeated per occurrence."""
    if ann.hapax_code:  # hapax -> judged in the absolute
        if ann.reference and normalise(ann.reference) != normalise(ann.target_canonical):
            return _md_escape(ann.reference) + " _(hint)_"
        return "- _(hapax: absolute judgment)_"
    return _md_escape(ann.reference or "(none)")


def _match_field(ann: OccurrenceAnnotation) -> str:
    """Match cell. Blank for hapax: the match does not route hapax occurrences
    (they bypass it and go straight to the hapax judge)."""
    if ann.hapax_code:
        return "-"
    return ann.match_level


def _q2_field(ann: OccurrenceAnnotation) -> str:
    # Q2 column: empty for hapax (Q2 not run on the hapax path)
    if ann.hapax_code:
        return "-"
    if ann.verdict in ERROR_VERDICTS or ann.match_level == "exact":
        return "-"
    return ann.q2_reason or "-"


def _render_legend(reference_mode: str = "") -> List[str]:
    """Reminder of all label sets used in the per-concept tables.

    Wrapped in a <details> block so it stays collapsed in markdown viewers
    that support GFM (VSCode preview, GitHub) and only takes screen space
    when the reader needs it. ``reference_mode`` adds a note about how the
    references R(s) were obtained (notably LLM-validated under ``validated``).
    """
    validated_note = []
    if "validated" in (reference_mode or ""):
        validated_note = [
            "**LLM-validated references**: under `--reference-mode validated`, each "
            "reference R(s) **outside the glossary** is validated by the error judge: "
            "kept if judged correct, otherwise **regenerated**. **Glossary (head term)** "
            "references are authoritative and are **not** validated. Provenance in "
            "`reference_source`: `glossary` = not validated; `frequency`/`first` = validated "
            "and kept; `generate` = heuristic rejected, then regenerated.",
            "",
        ]
    return [
        "<details>",
        "<summary><strong>Legend</strong>: Q1 / Q2 / verdict / match / var.cat codes and table columns (click to expand)</summary>",
        "",
        *validated_note,
        "**What the metric compares**",
        "",
        "For each term occurrence, the `Match` column compares **`Target canon`** (the canonical target form) with **`Reference R(s)`** (the reference selected by `--reference-mode`). The other columns are there for the human reader and for the LLM prompts.",
        "",
        "| Column | Meaning | Source |",
        "|---|---|---|",
        "| `Source (raw)` | source term as it appears in the segment (case preserved) | record `source_term` |",
        "| `Var.cat` | source variation category (see below) | record `variation_category` |",
        "| `1st?` | `*` if this is the first mention of the concept in the document | computed |",
        "| `Target raw` | translation span aligned in the system output (raw form) | record `target_term` |",
        "| `Target canon` | canonical target form, **compared with the reference** | record `target_canonical` |",
        "| `Reference R(s)` | reference for this **source form** (see the hapax note) | chosen by `--reference-mode` |",
        "| `Match` | result of the comparison `Target canon` vs `Reference R(s)`; `-` for hapax (not applicable) | computed |",
        f"| `Q1` | standard Q1 code (non-hapax): `{_SPAN}` (error), `NO`, or `-` if not applicable. Suffix `(from segN)`: verdict **propagated** from the occurrence of segment N (`--propagate-q1 on`), so NOT judged on this occurrence | LLM (if enabled) |",
        f"| `Q1 (hapax)` | code of the hapax judge: `{_SPAN}` (error) or `NO` (-> CONFORME) | LLM (if enabled) |",
        "| `Q2` | consistency label of the variation (non-hapax) | LLM |",
        f"| `Verdict` | final label (CONFORME, {_SPAN}, VAR_J, VAR_NJ) | derived from Q1/Q2 |",
        "| `Justification` | justification returned by the LLM (Q1 if error, Q2 otherwise), truncated | LLM |",
        "",
        "**Final verdict** (column `Verdict`)",
        "",
        "| Code | Meaning | Default penalty |",
        "|---|---|:---:|",
        "| `CONFORME` | target = R(s) after normalisation (exact or graphic), or judged error-free | 0 |",
        "| `VAR_J`    | justified variation (does communicative work / consistent with the document) | 0 |",
        f"| `VAR_NJ`   | unjustified variation (neutralisation, or gratuitous divergence) | {BASE_PENALTIES['VAR_NJ']:g} |",
        f"| `{_SPAN}`  | error (typology, see Q1) | {min(e.penalty for e in ERROR_TYPOLOGY.values()):g} to {max(e.penalty for e in ERROR_TYPOLOGY.values()):g} depending on the code |",
        "| `SKIPPED_HAPAX` | hapax excluded from scoring (`--hapax-mode skip`) | 0 |",
        "",
        "> **Hapax note**: for a hapax, the verdict comes from the hapax judge (**absolute** judgment), NOT from a match. `Match` is therefore `-`, and `Reference R(s)` is **not** a comparison target: it is only displayed (with `_(hint)_`) when it was actually given to the judge as a hint (i.e. different from the target); otherwise `- _(hapax: absolute judgment)_`.",
        "",
        "**Hapax handling** (single occurrence of a source form in the document)",
        "",
        f"- `--hapax-mode evaluate` (default): the hapax judge sees the chain of the **other source forms of the same concept** as evidence. Returns {_SPAN} (error) or NO (-> CONFORME).",
        "- `--hapax-mode skip`: verdict `SKIPPED_HAPAX`, penalty 0, not scored.",
        "- `--judge-path q1q2`: every alignment, hapax included, goes through Q1 then Q2 instead.",
        "",
        "**Q1 vs Q1 (hapax)**: two distinct paths depending on whether the occurrence is a hapax",
        "",
        f"- `Q1` is called on divergent **non-hapax** occurrences: compares the target with the reference and classifies {_SPAN} or NO. If NO, goes on to Q2.",
        f"- `Q1 (hapax)` is called on **every hapax**: judges the translation directly, with the concept chain (other forms) as evidence. Returns {_SPAN} (error) or NO (-> CONFORME). Q2 is NOT called afterwards.",
        "- An occurrence never uses both paths: it is one OR the other.",
        "",
        "**Q1: error vs variation classification** (central typology, `typology.py`; the prompt exposes a subset)",
        "",
        "| Code | Label | Gravity |",
        "|---|---|:---:|",
        *(
            f"| `{code}` | {entry.label} ({entry.legend}) | {entry.gravity} |"
            for code, entry in ERROR_TYPOLOGY.items()
        ),
        "| `NO` | not an error -> goes on to Q2 | - |",
        "",
        "**Q2: consistency judgment**: one LLM call, two axes: (negative) does the target erase a distinction made by the source? (positive) does using the target rather than R(s) do communicative work in context? The `Q2` column carries the returned label.",
        "",
        "| Label | Meaning | Verdict |",
        "|---|---|:---:|",
        "| `introduction`          | introduces the concept (first mention) | VAR_J |",
        "| `explicitation`         | first mention, expands or clarifies the term | VAR_J |",
        "| `avoid_repetition`      | avoids an awkward repetition in the same sentence | VAR_J |",
        "| `document_usage`        | follows the rendering already established for this concept in the document | VAR_J |",
        "| `register`              | adaptation to the register of the passage | VAR_J |",
        "| `facet`                 | highlights a legitimate facet of the concept | VAR_J |",
        "| `synonym_merge`         | merges two source forms whose distinction French cannot reproduce | VAR_J |",
        "| `neutralisation`        | target = R(s) of ANOTHER source form -> erases a distinction | VAR_NJ |",
        "| `synonym_inconsistency` | isolated switch to an admitted synonym, without function | VAR_NJ |",
        "| `gratuitous_divergence` | lexical divergence without function or licence | VAR_NJ |",
        "| `no_llm`                | LLM disabled (`--q2-mode skip` / dry-run) -> default | VAR_NJ |",
        "",
        "**Match level** (column `Match`)",
        "",
        "| Level | Meaning |",
        "|---|---|",
        "| `exact`   | strings equal after case / article normalisation -> direct CONFORME |",
        "| `graphic` | equal only after stripping hyphens/diacritics -> normalised to CONFORME (penalty 0) |",
        "| `none`    | no match -> divergence routed to Q1 |",
        "",
        "**Var.cat** (column `Var.cat`, source variation label relative to the head term)",
        "",
        "| Code | Meaning |",
        "|---|---|",
        "| `no_variation` | the source form IS the head term of the concept |",
        "| `VG`           | graphical variation (acronym, case, hyphens) |",
        "| `VR`           | reduction (element deleted, extension dropped) |",
        "| `VE`           | expansion (elements added) |",
        "| `VL`           | lexical variation (synonym) |",
        "| `VMS`          | morphosyntactic variation |",
        "| `CM`           | combined variation |",
        "",
        "**Note**: `Seg` cross-references the segment numbering in **Source text** / **Target text** below. Justifications are truncated to 140 characters.",
        "",
        "</details>",
        "",
    ]


def _render_full_text(title: str, lines: List[str]) -> List[str]:
    if not lines:
        return []
    out = [f"## {title}\n"]
    for i, line in enumerate(lines):
        out.append(f"{i:>3}. {line.strip()}")
    out.append("")
    return out


def render_doc_md(
    annotations: List[OccurrenceAnnotation],
    system: str,
    doc_id: str,
    src_lines: List[str] | None = None,
    tgt_lines: List[str] | None = None,
    run_tag: str = "",
    run_params: dict | None = None,
) -> str:
    """Return a markdown report for one document, one table per concept.

    If ``src_lines`` / ``tgt_lines`` are provided, they are rendered as
    numbered blocks at the top of the file so the segment indices in the
    per-concept tables can be cross-referenced.
    """
    if not annotations:
        return f"# {doc_id} ({system})\n\n_No annotations._\n"

    by_concept: Dict[str, List[OccurrenceAnnotation]] = defaultdict(list)
    for ann in annotations:
        by_concept[ann.concept_key].append(ann)

    total = len(annotations)
    n_concepts = len(by_concept)
    verdict_counts = Counter(a.verdict for a in annotations)

    out: List[str] = []
    out.append(f"# {doc_id} ({system})\n")
    if run_tag or run_params:
        out.append(f"**Run tag**: `{run_tag}`")
        if run_params:
            params_str = "  ".join(f"`{k}={v}`" for k, v in run_params.items())
            out.append(f"**Run params**: {params_str}")
        out.append("")
    out.append(f"- **Total occurrences:** {total}")
    out.append(f"- **Concepts:** {n_concepts}")
    out.append(f"- **Verdicts:** {_verdict_summary(verdict_counts)}")
    out.append("")

    # Reference mode (drives the R(s) provenance tags + the legend note)
    ref_mode = str((run_params or {}).get("reference_mode", ""))

    # Legend (collapsible)
    out.extend(_render_legend(ref_mode))

    # Full source/target texts at the top, with segment numbers matching the
    # `Seg` column in the per-concept tables below.
    if src_lines:
        out.append("---\n")
        out.extend(_render_full_text("Source text (EN)", src_lines))
    if tgt_lines:
        out.append("---\n")
        out.extend(_render_full_text(f"Target text (FR, {system})", tgt_lines))

    out.append("---\n")

    def concept_sort(item):
        ckey, anns = item
        first_seg = min((a.source_segment_ids[0] if a.source_segment_ids else 10**9) for a in anns)
        return (first_seg, ckey)

    for concept_key, anns in sorted(by_concept.items(), key=concept_sort):
        anns_sorted = sorted(anns, key=_sort_key)
        first = anns_sorted[0]

        out.append(f"## {concept_key} (term_id={first.term_id})\n")
        out.append(f"- **Source head term (prefLabel EN)**: `{first.concept_preflabel_en or '(not in glossary)'}`")
        out.append(f"- **Target head term (prefLabel FR)**: `{first.concept_preflabel_fr or '(not in glossary)'}`")
        out.append(f"- **Occurrences**: {len(anns_sorted)}")
        sub_dist = Counter(a.verdict for a in anns_sorted)
        out.append(f"- **Verdicts**: {_verdict_summary(sub_dist)}")
        # Distinct (src_form -> R(s)) pairs: multiple if the concept has
        # several source forms (e.g. head term + acronym), each gets its own R.
        ref_pairs = sorted({
            (a.source_term.strip().lower(), a.reference or "(none)", a.reference_source)
            for a in anns_sorted
        })

        def _prov(src_kind: str) -> str:
            # Provenance of R(s). Under validated, show the interpretation
            # (validated / regenerated / glossary) next to the raw source.
            tag = _ref_source_tag(src_kind, ref_mode)
            if "validated" in ref_mode and tag:
                return f"source: `{src_kind}` -> {tag}"
            return f"source: `{src_kind or '-'}`"

        if len(ref_pairs) == 1:
            _src, _ref, _kind = ref_pairs[0]
            out.append(f"- **Reference R(s)**: `{_ref}` ({_prov(_kind)})")
        else:
            out.append(f"- **Distinct references R(s)** for this concept ({len(ref_pairs)}):")
            for src, ref, src_kind in ref_pairs:
                out.append(f"    - `{src}` -> `{ref}` ({_prov(src_kind)})")
        out.append("")

        # Table: comparison columns are ordered to read left-to-right, source
        # side first, then target side, then the comparison.
        out.append(
            "| Seg | Source (raw) | Var.cat | 1st? | Target raw | Target canon | Reference R(s) | Match | Q1 | Q1 (hapax) | Q2 | Verdict | Justification |"
        )
        out.append(
            "|----:|--------------|---------|:----:|------------|--------------|----------------|-------|----|------------|----|---------|---------------|"
        )
        for ann in anns_sorted:
            seg = ann.source_segment_ids[0] if ann.source_segment_ids else ""
            tgt_raw = ann.target_term or ann.target_canonical
            row = "| {seg} | {src} | {cat} | {first} | {tgt_raw} | {tgt_can} | {ref} | {lvl} | {q1} | {hap} | {q2} | **{verdict}** | {justif} |".format(
                seg=seg,
                src=_md_escape(ann.source_term),
                cat=_md_escape(ann.variation_category) or "-",
                first="*" if ann.is_first_mention else "",
                tgt_raw=_md_escape(tgt_raw),
                tgt_can=_md_escape(ann.target_canonical),
                ref=_ref_field(ann),
                lvl=_match_field(ann),
                q1=_md_escape(_q1_field(ann)),
                hap=_md_escape(_hapax_field(ann)),
                q2=_md_escape(_q2_field(ann)),
                verdict=ann.verdict,
                justif=_md_escape(_truncate(_justif_field(ann))),
            )
            out.append(row)
        out.append("")

    return "\n".join(out) + "\n"


def write_chain_md(
    path: Path,
    annotations: List[OccurrenceAnnotation],
    system: str,
    doc_id: str,
    src_lines: List[str] | None = None,
    tgt_lines: List[str] | None = None,
    run_tag: str = "",
    run_params: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_doc_md(
            annotations,
            system=system,
            doc_id=doc_id,
            src_lines=src_lines,
            tgt_lines=tgt_lines,
            run_tag=run_tag,
            run_params=run_params,
        ))

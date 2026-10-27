"""Q2: qualify a non-error divergence (Q1 returned NO) as JUSTIFIED or not.

Q1 has established that the target denotes the correct concept but differs
from the expected reference R(s). Q2 makes a single judgment, with one LLM
call, on document consistency:

  - negative axis (does it destroy a source distinction?): the target
    reproduces the reference of ANOTHER source form of the same concept,
    erasing a distinction the author kept (neutralisation);
  - positive axis (does it do communicative work?): introducing/explicitating
    the concept, avoiding an awkward repetition, following the rendering the
    document already established for this concept, adapting to register, or
    foregrounding a legitimate facet.

Verdict: ``VAR_NJ`` (not justified) if it erases a distinction OR does no
work; ``VAR_J`` (justified) otherwise. A ``reason`` label feeds the
qualitative profile. No deterministic rule is applied here: the whole
judgment is the LLM's (the "one LLM call, nothing else" design).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ..io.skos import GlossaryEntry
from ..llm.client import SALVAGED_NOTE, LLMClient, extract_json, salvage_labels
from .context import build_discourse_context
from .match import any_equivalence

# Labels of the Q2 prompt meaning "justified variation" (VAR_J); the three
# inconsistency labels (neutralisation, synonym_inconsistency,
# gratuitous_divergence) and any unknown label mean VAR_NJ. Used when the
# output schema carries no "justified" boolean.
_COHERENT_LABELS = frozenset({
    "introduction", "explicitation", "avoid_repetition", "document_usage",
    "register", "facet", "synonym_merge",
})
from .variation import develop as develop_variation


VAR_CODES = {"VAR_J", "VAR_NJ"}

# Reason labels emitted by the LLM, grouped by the verdict they imply. Kept
# here so downstream code (profiling, validation) can map a label to a verdict.
JUSTIFIED_LABELS = {
    "introduction", "explicitation", "avoid_repetition",
    "document_usage", "register", "facet", "synonym_merge",
}
NOT_JUSTIFIED_LABELS = {
    "neutralisation", "synonym_inconsistency", "gratuitous_divergence",
}


@dataclass
class Q2Result:
    code: str                     # VAR_J | VAR_NJ
    reason: str = ""              # LLM label (see *_LABELS sets above)
    justification: str = ""


def _format_other_refs(other_form_references: Dict[str, str]) -> str:
    if not other_form_references:
        return "  (no other source form of this concept is known in the document)"
    return "\n".join(
        f"  - « {src} » → « {ref} »"
        for src, ref in sorted(other_form_references.items())
    )


def qualify(
    *,
    target_canonical: str,
    reference: str,
    src_term: str,
    src_form_raw: str,
    tgt_aligned_raw: str,
    src_canonical: str,
    src_segment: str,
    tgt_segment: str,
    variation_category: str,
    other_form_references: Dict[str, str],
    glossary_entry: Optional[GlossaryEntry] = None,
    is_first_mention: bool = False,
    concept_history: str = "",
    context_window: str = "",
    use_llm: bool = True,
    llm: Optional[LLMClient] = None,
    llm_prompt_tmpl: Optional[str] = None,
    dump_label: str = "",
    concept_renderings: str = "",
    context_pairs: str = "",
    head_or_variant: str = "",
) -> Q2Result:
    """One LLM judgment.

    Without an LLM (``use_llm`` False / dry-run / no client) the divergence
    cannot be judged, so it defaults to ``VAR_NJ`` (a divergence we cannot
    justify).
    """
    if not use_llm or llm is None or llm_prompt_tmpl is None:
        return Q2Result(
            code="VAR_NJ",
            reason="no_llm",
            justification="No LLM judgment available (defaulting to not justified).",
        )

    alt_labels: list = []
    if glossary_entry is not None:
        try:
            alt_labels = [a for a in glossary_entry.all_fr() if a]
        except Exception:
            alt_labels = []
    is_admitted = bool(alt_labels) and any_equivalence(target_canonical, alt_labels)

    # Glossary head term for the concept: the authoritative denomination
    # against which this source form's reference and the other forms' references
    # are variants. Shown to the judge as terminological context.
    concept_preflabel_fr = ""
    if glossary_entry is not None:
        concept_preflabel_fr = getattr(glossary_entry, "preflabel_fr", "") or ""

    # When the template has a ``{concept_renderings}`` block, that block already
    # carries every rendering of the concept (per source form), so the discourse
    # context keeps only the preceding segments PAIRED EN/FR, to avoid
    # duplicating the information. Other templates get the full construction.
    if "{concept_renderings}" in (llm_prompt_tmpl or ""):
        discourse_context = (
            ("Discourse context — preceding segments (each EN line paired with its FR "
             "translation, most recent last):\n" + context_pairs.strip("\n") + "\n\n")
            if (context_pairs or "").strip() else ""
        )
    else:
        discourse_context = build_discourse_context([
            ("Preceding segments", context_window),
            ("Prior occurrences of this concept in the document (each line: the source form and its translation by the system)", concept_history),
            ("Reference translations of the OTHER source forms of the same concept",
             _format_other_refs(other_form_references) if other_form_references else ""),
        ])

    # Composite "Concept" line of the Q2 template: EN + FR head terms and the
    # admitted glossary synonyms, on one line; the is_admitted question is
    # phrased as "one of the French glossary designations above?".
    if glossary_entry is not None and (concept_preflabel_fr
                                       or getattr(glossary_entry, "preflabel_en", "")):
        _en = getattr(glossary_entry, "preflabel_en", "") or "(unknown)"
        _fr = f'"{concept_preflabel_fr}"' if concept_preflabel_fr else "(none listed)"
        _alts = [a for a in getattr(glossary_entry, "altlabels_fr", []) or [] if a]
        _alts_txt = ", ".join(f'"{a}"' for a in _alts) if _alts else "(none listed)"
        concept_glossary_line = (
            f'Concept: "{_en}" (English head term). Glossary French designation: '
            f"{_fr}; admitted French synonyms: {_alts_txt}."
        )
        is_glossary_designation = "yes" if is_admitted else "no"
    else:
        concept_glossary_line = ("Concept: (not in the glossary — no French designation "
                                 "or admitted synonyms are listed).")
        is_glossary_designation = "(not applicable — concept not in the glossary)"
    user = llm_prompt_tmpl.format(
        src_form_raw=src_form_raw or "(empty)",
        tgt_aligned_raw=tgt_aligned_raw or "(empty)",
        src_canonical=src_canonical or "(empty)",
        tgt_canonical=target_canonical or "(empty)",
        reference=reference,
        concept_preflabel_fr=concept_preflabel_fr or "(not in glossary)",
        src_segment=src_segment or "(none)",
        tgt_segment=tgt_segment or "(none)",
        variation_category=develop_variation(variation_category),
        concept_history=concept_history or "(first occurrence in document)",
        context_window=context_window or "  (none)",
        other_form_references=_format_other_refs(other_form_references),
        is_admitted="yes" if is_admitted else "no",
        is_first_mention="yes" if is_first_mention else "no",
        discourse_context=discourse_context,
        concept_renderings=concept_renderings,
        head_or_variant=head_or_variant or "(unknown)",
        concept_glossary_line=concept_glossary_line,
        is_glossary_designation=is_glossary_designation,
    )
    response = llm.chat(
        prompt_name="q2_vartj",
        system=(
            "You are an expert terminologist judging whether a non-error "
            "terminological variation is justified in context."
        ),
        user=user,
        dump_label=dump_label or None,
    )
    parsed = extract_json(response)
    if not isinstance(parsed, dict):
        # Malformed JSON with an intact label field: keep the label if it is
        # the only one found and a known reason label.
        labels = {l.lower() for l in salvage_labels(response)}
        if len(labels) != 1 or not labels <= JUSTIFIED_LABELS | NOT_JUSTIFIED_LABELS:
            return Q2Result(
                code="VAR_NJ",
                reason="unparseable",
                justification="LLM output unparseable (defaulting to not justified).",
            )
        parsed = {"label": labels.pop(), "reason": SALVAGED_NOTE}
    label = str(parsed.get("label", "") or "").strip()
    reason = str(parsed.get("reason") or parsed.get("justification") or "")
    if "justified" in parsed:
        # A boolean emitted by the model takes precedence.
        justified = bool(parsed.get("justified"))
    else:
        # No boolean in the schema: the label decides (coherent -> VAR_J;
        # inconsistency -> VAR_NJ; unknown -> VAR_NJ).
        justified = label.lower() in _COHERENT_LABELS
    code = "VAR_J" if justified else "VAR_NJ"
    if not label:
        label = "justified" if justified else "not_justified"
    return Q2Result(code=code, reason=label, justification=reason)

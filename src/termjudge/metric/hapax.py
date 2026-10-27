"""LLM judgement of hapax source forms (and of candidate references).

A hapax is a (concept, source form) chain with a single occurrence in the
document. The standard Q1/Q2 path would compare the target against itself
(frequency fallback), trivially yielding CONFORME and silently masking
potential errors. :func:`judge` therefore asks the LLM to evaluate the
translation in absolute terms, with the Q1 prompt text, using as evidence
the chain of OTHER occurrences of the same concept (other source forms) in
the document.

The same function validates a candidate pseudo-reference under
``--reference-mode validated`` (see ``reference_selector``).

Return: an error code (typology.py) or ``NO`` (acceptable; the caller maps
it to ``CONFORME``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..llm.client import LLMClient
from .context import build_discourse_context
from .q1 import (PRECEDING_SEGMENTS_HEADER, PRIOR_RENDERINGS_HEADER, Q1_CODES,
                 Q1_LABEL_TO_CODE, _concept_block, _section)
from .variation import develop as develop_variation


# Default system message: it is part of the CACHE KEY, so it must never change
# by a byte. Any caller that does not override ``system`` finds its existing
# cache entries.
_DEFAULT_SYSTEM = ("You are an expert terminologist judging the quality of an "
                   "in-context translation for a hapax term form.")

# Override for reference validation (--reference-mode validated) when the
# candidate is judged from source-side evidence only (no occurrence to show).
VALIDATE_SYSTEM = ("You are an expert terminologist judging whether a candidate "
                   "French reference form is a valid designation of an English "
                   "source term, from source-side evidence only.")

# Header of the ``concept_chain`` section. By default (judging a hapax) it
# contains translations; in source-only reference validation it contains only
# SOURCE segments, hence the alternative header. The default is frozen byte for
# byte (it enters the prompt text, hence the cache key).
_DEFAULT_CHAIN_LABEL = ("How the same concept is translated elsewhere in the document "
                        "(other source forms and their translations)")
VALIDATE_CHAIN_LABEL = ("Other English segments of this document where this source form occurs "
                        "(SOURCE side only — no translation is shown here)")


@dataclass
class HapaxResult:
    code: str            # error code | NO
    justification: str = ""


def judge(
    llm: LLMClient,
    prompt_tmpl: str,
    src_segment: str,
    tgt_segment: str,
    src_form_raw: str,
    tgt_aligned_raw: str,
    src_canonical: str,
    tgt_canonical: str,
    concept_preflabel_en: str,
    concept_preflabel_fr: str,
    variation_category: str,
    concept_chain: str,
    reference: Optional[str] = None,
    context_window: str = "",
    dump_label: str = "",
    dump_name: str = "hapax_judge",
    system: Optional[str] = None,
    chain_label: Optional[str] = None,
    prompt_name: str = "hapax_judge",
    context_pairs: str = "",
    concept_history_indexed: str = "",
) -> HapaxResult:
    """Judge a hapax translation, optionally with an expected reference.

    When ``reference`` is provided (e.g. the source form has a glossary
    translation), templates that reference it ask the LLM to treat it as the
    expected target. Otherwise the LLM judges on its own merits, using the
    ``concept_chain`` as evidence of in-document usage.
    """
    if reference:
        reference_block = (
            f"Expected reference for this source form (from glossary or "
            f"frequency analysis): {reference}"
        )
    else:
        reference_block = (
            "Expected reference: (none — no glossary translation and no "
            "in-document chain to derive one from)"
        )
    # Single conditional bullet, omitted entirely when there is no reference
    # (rather than printing an "(none)" placeholder line).
    expected_reference = (
        f"\n- Expected reference for this English term: {reference}" if reference else ""
    )
    discourse_context = build_discourse_context([
        ("Preceding segments", context_window),
        (chain_label or _DEFAULT_CHAIN_LABEL, concept_chain),
    ])
    user = prompt_tmpl.format(
        # Blocks of the Q1 template (the hapax judge reuses the Q1 prompt).
        # Rendered by the q1 functions (same headers, same auto-collapse), so a
        # Q1 template yields here exactly the text it would yield through
        # q1.classify. Empty by default -> sections absent.
        concept_block=_concept_block(concept_preflabel_en, concept_preflabel_fr),
        preceding_segments=_section(PRECEDING_SEGMENTS_HEADER, context_pairs),
        prior_renderings=_section(PRIOR_RENDERINGS_HEADER, concept_history_indexed),
        src_segment=src_segment,
        tgt_segment=tgt_segment,
        src_form_raw=src_form_raw or "(empty)",
        tgt_aligned_raw=tgt_aligned_raw or "(empty)",
        src_canonical=src_canonical or "(empty)",
        tgt_canonical=tgt_canonical or "(empty)",
        concept_preflabel_en=concept_preflabel_en or "(not in glossary)",
        concept_preflabel_fr=concept_preflabel_fr or "(not in glossary)",
        variation_category=develop_variation(variation_category),
        concept_chain=concept_chain or "(no other occurrences of this concept)",
        reference_block=reference_block,
        expected_reference=expected_reference,
        context_window=context_window or "  (none)",
        discourse_context=discourse_context,
    )
    parsed = llm.chat_json(
        prompt_name=prompt_name,     # CACHE key
        system=system or _DEFAULT_SYSTEM,   # in the CACHE key
        user=user,
        dump_label=dump_label or None,
        dump_name=dump_name,         # DUMP folder (may differ)
    )
    if not isinstance(parsed, dict):
        return HapaxResult(code="NO", justification="LLM output unparseable (defaulting to NO)")
    # "code" (a bare code) or "label" (the mnemonic label of the Q1 schema,
    # mapped to its code).
    raw = str(parsed.get("code") or parsed.get("label") or "").strip()
    code = Q1_LABEL_TO_CODE.get(raw.lower(), raw.upper())
    if code not in Q1_CODES:
        return HapaxResult(code="NO", justification=f"Invalid code {raw!r}, defaulting to NO")
    return HapaxResult(code=code, justification=parsed.get("justification", "") or "")

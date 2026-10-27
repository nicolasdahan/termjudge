"""Q1: classify a divergence as an error code (typology.py) or NO (variation).

One LLM call per divergence. The valid error codes come from the central
registry ``typology.ERROR_TYPOLOGY`` (the superset typology); the prompt may
expose only a subset of them to the LLM. Expected output: JSON
``{justification, label}``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..llm.client import SALVAGED_NOTE, LLMClient, extract_json, salvage_labels
from .context import build_discourse_context
from .typology import ERROR_TYPOLOGY
from .variation import develop as develop_variation


Q1_CODES = set(ERROR_TYPOLOGY) | {"NO"}

# Mnemonic labels exposed by the Q1 prompt: the judge answers with a readable
# LABEL rather than an opaque code (fewer copy errors than with "A6" vs "A7").
# Minimal post-processing: the label is mapped back to the canonical code
# before validation.
Q1_LABEL_TO_CODE = {
    "wrong_related_term": "A1",
    "paraphrase_not_term": "A2",
    "invented_calque": "A3",
    "left_untranslated": "A4",
    "dnt_violated": "A5",
    "wrong_component": "A6",
    "general_word_not_term": "A7",
    "misparsed_structure": "C",
    "wrong_constituent_relation": "D",
    "failed_transposition": "E",
    "grammar_phraseology": "F",
    "content_altered": "G",
    "defective_register_form": "H",
    "no_error": "NO",
}


def _section(header: str, body: str) -> str:
    """Optional "header + body" section, dropped entirely when the body is
    empty (no phantom heading, no '(none)'). Ends with a blank line so the
    next section follows cleanly."""
    body = (body or "").strip("\n")
    return f"{header}\n{body}\n\n" if body.strip() else ""


# Headers of the supporting sections of the Q1 template. Named so that
# hapax.py renders the SAME text when the hapax judge reuses the Q1 prompt:
# they enter the prompt, hence the cache key, and must never diverge by a byte
# between the two callers.
# Q1 system message: part of the CACHE KEY, frozen byte for byte. Named so that
# pseudo-reference validation (same prompt) sends exactly the same one.
Q1_SYSTEM = ("You are an expert terminologist annotating translation errors "
             "according to a fixed typology.")

PRECEDING_SEGMENTS_HEADER = ("Discourse context — preceding segments (each EN line paired with its "
                             "FR translation, most recent last):")
PRIOR_RENDERINGS_HEADER = ("Prior renderings of this concept in the document (THIS SYSTEM's own earlier "
                           "outputs — support only, never attested usage; a repeated rendering is not "
                           "thereby correct):")



# What "expected form" evidence Q1 shows the judge. Q1's core question is
# absolute ("is the target a valid translation of the source term?"); the
# anchor is only supporting evidence. The published Q1 prompt builds its own
# evidence from the concept block and does not use these anchor blocks; they
# are kept for templates that reference ``{anchor_block}`` / ``{evidence_block}``.
Q1_ANCHOR_MODES = ("none", "reference", "vedette", "both")


# Wording shared by the two anchor builders below (``build_anchor_block`` and
# the provenance-aware ``build_evidence_block``): one source of truth.
_NO_ANCHOR_LINE = (
    "No expected form is provided; judge the translation of the source term "
    "on its own merits and against the concept's other occurrences below."
)


def _concept_block(concept_preflabel_en: str, concept_preflabel_fr: str) -> str:
    """CONCEPT block (concept level, stable): the glossary head term, an
    authoritative and system-independent piece of evidence, distinct from
    R(s), which belongs to ONE source form and has a variable provenance.
    The section is dropped when the concept is not in the glossary. Same
    pattern as ``hapax.judge``."""
    lines = []
    if (concept_preflabel_en or "").strip():
        lines.append(f"- Concept head term (English): {concept_preflabel_en}")
        if (concept_preflabel_fr or "").strip():
            lines.append(f"- Established French translation of this head term (from the "
                         f"glossary): {concept_preflabel_fr}")
    elif (concept_preflabel_fr or "").strip():
        # No English head term to display: "this head term" would refer to nothing.
        lines.append(f"- Established French term for this concept (from the glossary): "
                     f"{concept_preflabel_fr}")
    return ("Concept:\n" + "\n".join(lines) + "\n\n") if lines else ""


def _vedette_line(concept_preflabel_fr: str) -> str:
    """"Glossary head term" line (or its absence)."""
    if concept_preflabel_fr:
        return (
            "Established term for this concept (glossary head term): "
            "{ved}. Use it as evidence of the expected terminology; there is not "
            "necessarily a single mandatory form.".format(ved=concept_preflabel_fr)
        )
    return (
        "Established term for this concept: (not in glossary — judge on the "
        "source term's merits and the concept's other occurrences)."
    )


def build_anchor_block(mode: str, reference: str, concept_preflabel_fr: str) -> str:
    """Render the ``{anchor_block}`` of a Q1 template.

    - ``none``      : no expected form; judge on the term's merits + concept history.
    - ``reference`` : the per-source-form R(s) (heuristic, possibly imperfect).
    - ``vedette``   : the glossary head term for the concept (authoritative,
                      system-independent); falls back to a "not in glossary"
                      note when absent.
    - ``both``      : vedette then reference.
    """
    ref_line = (
        "Expected reference for this source form (a working reference selected "
        "heuristically from a glossary or the document's most frequent / first "
        "rendering, and possibly imperfect): {ref}".format(ref=reference or "(none)")
    )
    ved_line = _vedette_line(concept_preflabel_fr)
    none_line = _NO_ANCHOR_LINE
    if mode == "none":
        return none_line
    if mode == "reference":
        return ref_line
    if mode == "both":
        return ved_line + "\n" + ref_line
    # default / "vedette"
    return ved_line

def build_evidence_block(
    mode: str,
    reference: str,
    reference_source: str,
    match_level: str,
    concept_preflabel_fr: str,
    target_canonical: str,
    validated_refs: bool = False,
) -> str:
    """Render the ``{evidence_block}``: the anchor as a FUNCTION of the
    provenance of R(s) and of its relation to the target, not a constant line.

    Six states, because a reference does not have the same evidential value
    depending on where it comes from, and has NONE when it is the target
    itself (frequency fallback on a single-occurrence form):

    - ``none``        : anchoring disabled;
    - absent          : no expected form available;
    - self-reference  : R(s) == the judged target -> no evidence, absolute judgment;
    - authoritative   : glossary head term;
    - validated       : heuristic confirmed by an independent judge;
    - regenerated     : produced from the concept + SOURCE contexts;
    - heuristic       : most frequent form of the document, possibly from the
                        MT output itself (circularity) -> weak evidence.
    """
    from .match import normalise as _norm

    if mode == "none":
        return ("- Expected form for this source form: (not provided) — judge the target on its "
                "own merits and against the concept's other occurrences above.")
    if not reference:
        return ("- Expected form for this source form: NONE could be established — judge the "
                "target on its own merits and against the concept's other occurrences above.")
    if _norm(reference) == _norm(target_canonical or ""):
        return ("- Expected form for this source form: NO INDEPENDENT ONE — the only rendering "
                "observed for this form in the document IS the target under evaluation, so it "
                "cannot serve as evidence. Judge the target on its own merits; do NOT treat this "
                "coincidence as confirmation.")

    if reference_source == "glossary":
        line = f"- Expected form for this source form (glossary, authoritative): {reference}"
    elif reference_source == "generate":
        line = (f"- Expected form for this source form (generated from the concept and its "
                f"SOURCE-side contexts, never from system output): {reference}")
    elif validated_refs and reference_source in ("frequency", "first"):
        line = (f"- Expected form for this source form (observed in this document, then "
                f"independently validated as correct terminology): {reference}")
    else:
        line = (f"- Expected form for this source form (heuristic: the most frequent/first "
                f"rendering observed IN THIS DOCUMENT — it may itself come from machine "
                f"translation, so it is weak evidence): {reference}")
    return line + ("\n- Treat it as EVIDENCE, not a verdict: a target that differs from it can "
                   "still be correct, and a target that matches it can still be wrong.")


# Gravity levels from the central registry. NOTE: membership in this dict is
# what routes a Q1 code to the "error" branch in annotate.py; it is NOT the
# classifier's tie-break order. The Q1 prompt breaks ties between co-occurring
# codes by SPECIFICITY of the error mechanism, not by severity, so that the
# classifier does not double-count severity already encoded here.
Q1_GRAVITY = {code: entry.gravity for code, entry in ERROR_TYPOLOGY.items()}


@dataclass
class Q1Result:
    code: str
    justification: str = ""
    raw: str = ""


def classify(
    llm: LLMClient,
    prompt_tmpl: str,
    src_segment: str,
    tgt_segment: str,
    src_form_raw: str,
    tgt_aligned_raw: str,
    src_canonical: str,
    tgt_canonical: str,
    reference: str,
    variation_category: str,
    concept_history: str,
    context_window: str = "",
    concept_preflabel_fr: str = "",
    anchor_mode: str = "vedette",
    dump_label: str = "",
    reference_source: str = "",
    match_level: str = "",
    validated_refs: bool = False,
    concept_preflabel_en: str = "",
    context_pairs: str = "",
    concept_history_indexed: str = "",
) -> Q1Result:
    anchor_block = build_anchor_block(anchor_mode, reference, concept_preflabel_fr)
    # Every block the template family may reference is rendered; str.format
    # ignores the ones the loaded template does not use.
    discourse_context = build_discourse_context([
        ("Preceding segments", context_window),
        ("Prior occurrences of this concept in the document (each line: the source form and its translation by the system)", concept_history),
    ])
    user = prompt_tmpl.format(
        src_segment=src_segment,
        tgt_segment=tgt_segment,
        src_form_raw=src_form_raw or "(empty)",
        tgt_aligned_raw=tgt_aligned_raw or "(empty)",
        src_canonical=src_canonical or "(empty)",
        tgt_canonical=tgt_canonical or "(empty)",
        reference=reference,
        anchor_block=anchor_block,
        # SEPARATE anchors, each placed next to what it qualifies (as in the
        # hapax judge): the concept (head term, stable and authoritative) on one
        # side, R(s) (per source form, variable provenance) on the other.
        concept_block=_concept_block(concept_preflabel_en, concept_preflabel_fr),
        # Separate sections, each with its caveat AT THE POINT OF USE (more
        # reliable than stating it far upstream in the instructions).
        preceding_segments=_section(PRECEDING_SEGMENTS_HEADER, context_pairs),
        prior_renderings=_section(PRIOR_RENDERINGS_HEADER, concept_history_indexed),
        evidence_block=build_evidence_block(
            anchor_mode, reference, reference_source, match_level,
            concept_preflabel_fr, tgt_canonical, validated_refs,
        ),
        variation_category=develop_variation(variation_category),
        concept_history=concept_history or "(first occurrence in document)",
        context_window=context_window or "  (none)",
        discourse_context=discourse_context,
    )
    response = llm.chat(
        prompt_name="q1_classify",
        system=Q1_SYSTEM,
        user=user,
        dump_label=dump_label or None,
    )
    parsed = extract_json(response)
    if not isinstance(parsed, dict):
        # Malformed JSON with an intact label field: keep the label if every
        # one found maps to the same valid code.
        codes = {Q1_LABEL_TO_CODE.get(l.lower(), l.upper()) for l in salvage_labels(response)}
        if len(codes) != 1 or not codes <= Q1_CODES:
            return Q1Result(code="NO", justification="LLM output unparseable (defaulting to NO)")
        parsed = {"label": codes.pop(), "justification": SALVAGED_NOTE}
    # The prompt answers under the key "label" (consistent with its instruction
    # "answer with exactly ONE label"); "code" is accepted as well.
    raw = str(parsed.get("label") or parsed.get("code") or "").strip()
    # Post-processing: the prompt makes the judge answer with a mnemonic label
    # ("wrong_component"); map it to the canonical code. A code emitted directly
    # passes unchanged.
    code = Q1_LABEL_TO_CODE.get(raw.lower(), raw.upper())
    if code not in Q1_CODES:
        return Q1Result(code="NO", justification=f"Invalid code {raw!r}, defaulting to NO")
    return Q1Result(code=code, justification=parsed.get("justification", "") or "")

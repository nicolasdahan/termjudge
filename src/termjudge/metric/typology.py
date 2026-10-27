"""Central registry of the terminology error typology.

SINGLE SOURCE OF TRUTH for the Q1 error codes. Everything else derives from
it: ``Q1_CODES`` and ``Q1_GRAVITY`` (q1.py), ``DEFAULT_PENALTIES``
(scoring.py), ``VERDICTS`` (annotation.py), the display order and the legend
(chains.py).

The registry holds the most complete typology (the superset). The prompts
expose only a subset of these codes to the LLM: codes that are not offered
are simply never emitted, and the downstream chain (validation, scoring,
aggregation, chains) stays correct without further changes.

Adding a category is ONE line here plus the prompt that describes it. A code
emitted by the LLM but absent from this registry is mapped to NO (see q1.py
and hapax.py), so never introduce a code in a prompt without registering it
here first.

Insertion order matters: it fixes the order of columns and legend tables.
"""

from __future__ import annotations

from typing import Dict, NamedTuple


class ErrorCode(NamedTuple):
    label: str      # short mnemonic (for human reading)
    gravity: int    # gravity (level A of the typology), for documentation and sorting
    penalty: float  # default scoring penalty (override with --penalties)
    legend: str     # short description for the chains legend


# Superset aligned on the error typology of Carcamo et al. Gravities follow the
# "Score" column of the typology (scale of Minder et al., 2024); penalties for
# the fine-grained codes follow the mapping gravity 1 -> 1.0, 2 -> 3.0, 3 -> 4.0.
# Deliberate exclusions:
#   - B1..B7 (in-document terminology inconsistency): already covered by the Q2
#     axis (VAR_J / VAR_NJ: neutralisation, synonym_inconsistency, ...); listing
#     them here would count the same thing twice;
#   - I (no error): that is the NO code;
#   - J (annotator type): a human-workflow label, irrelevant to an LLM judge.
ERROR_TYPOLOGY: Dict[str, ErrorCode] = {
    # A. Incorrect selection of the terminological equivalent
    "A1": ErrorCode("term_inexact",        1, 1.0, "close but inexact term"),
    "A2": ErrorCode("term_paraphr",        1, 1.0, "paraphrase instead of the term"),
    "A3": ErrorCode("creation_litt",       2, 3.0, "calque not used in the domain"),
    "A4": ErrorCode("non-trad",            3, 4.0, "source term left untranslated"),
    "A5": ErrorCode("DNT_traduit",         3, 4.0, "do-not-translate item translated"),
    "A6": ErrorCode("err_constituant",     3, 4.0, "wrong internal constituent"),
    "A7": ErrorCode("mot_langue_generale", 2, 3.0, "general-language word instead of the term"),
    # C. Constituents (syntactic analysis) of complex terms / noun phrases
    "C1": ErrorCode("tete_erronee",        3, 4.0, "wrong identification of the head of the term"),
    "C2": ErrorCode("modifieur_rattache",  3, 4.0, "modifier attached to the wrong constituent"),
    "C3": ErrorCode("ordre_constituants",  3, 4.0, "wrong order of constituents"),
    # D. Semantic relations
    "D1": ErrorCode("liens_semantiques",   3, 4.0, "wrong rendering of the semantic links between constituents"),
    "D2": ErrorCode("factorisation",       3, 4.0, "factorised constituent not detected in a coordination"),
    # E. Transposition
    "E1": ErrorCode("explicitation_absente", 2, 3.0, "synthetic structure not adapted to the analytic one"),
    "E2": ErrorCode("calque_syntaxique",   1, 1.0, "syntactic calque of the English structure"),
    # F. Grammatical / phraseological errors tied to the context
    "F1": ErrorCode("determination_accord", 1, 1.0, "determiner, or gender/number agreement"),
    "F2": ErrorCode("preposition",         2, 3.0, "missing or wrong preposition"),
    "F3": ErrorCode("collocation",         2, 3.0, "collocation/phraseology wrong for the domain"),
    "F4": ErrorCode("orthographe",         2, 3.0, "spelling error"),
    # G. Content alteration
    "G1": ErrorCode("ajout_injustifie",    3, 4.0, "unjustified addition of a lexical unit or phrase"),
    "G2": ErrorCode("omission",            3, 4.0, "omission of one or more constituents"),
    "G3": ErrorCode("hallucination",       3, 4.0, "unintelligible translation or hallucination"),
    # H. Target-language expression
    "H1": ErrorCode("poids_constituants",  2, 3.0, "mishandled weighting of constituents (punctuation, hierarchy)"),
    "H2": ErrorCode("registre",            1, 1.0, "style/register inappropriate for the target text"),
    # GROUPED codes, one per family (the codes the Q1 prompt exposes next to
    # A1..A7). They coexist in the superset with the fine-grained codes above;
    # a prompt exposes one set or the other. A grouped verdict asserts an error
    # of the family without identifying the sub-code, and the sub-codes of a
    # family differ in gravity: each family is therefore scored at its most
    # lenient sub-code (benefit of the doubt). Appended at the END of the
    # registry so that the column/legend order of the fine-grained codes is
    # unchanged.
    "C": ErrorCode("constituants_synt",    3, 4.0, "constituents of complex terms (head, attachment, order)"),
    "D": ErrorCode("relations_sem",        3, 3, "semantic relations between constituents (links, factorisation)"),
    "E": ErrorCode("transposition",        2, 1.0, "synthetic-to-analytic transposition / syntactic calque"),
    "F": ErrorCode("gram_phraseo",         2, 1.0, "grammar/phraseology in context (agreement, preposition, collocation, spelling)"),
    "G": ErrorCode("alteration_contenu",   3, 1.0, "content alteration (addition, omission, hallucination)"),
    "H": ErrorCode("expression_cible",     2, 1.0, "target-language expression/register (weighting of constituents, register)"),
}

# Error codes in the canonical order of the registry.
ERROR_CODES: tuple = tuple(ERROR_TYPOLOGY)

# Non-error verdicts (produced by the pipeline, not by the LLM) and their penalties.
BASE_PENALTIES: Dict[str, float] = {
    "CONFORME": 0.0,
    "VAR_J": 0.0,          # justified variation: no penalty
    "VAR_NJ": 0.5,         # unjustified variation (neutralisation / gratuitous)
    "SKIPPED_HAPAX": 0.0,  # excluded from scoring; counted separately
}


def codes_span() -> str:
    """Compact rendering of the superset for prose (e.g. ``A1..H``)."""
    return f"{ERROR_CODES[0]}..{ERROR_CODES[-1]}" if ERROR_CODES else "(none)"

"""Choose the reference translation R(s) for each (concept, source form, doc).

A reference is picked once per (concept_key, source_form, doc_id).

CAVEAT: the reference is NOT shared across MT systems. ``annotate.annotate_doc``
builds its observations from a SINGLE system, so only the ``glossary``
strategy (the head term's prefLabel_fr) and a ``generate`` result (source-side
prompt, hence a shared cache entry) are identical for every system. The
``frequency`` / ``first`` strategies are computed from the judged system's OWN
outputs, and under ``validated`` the decision to regenerate depends on that
system's output too. Two systems can therefore be scored against DIFFERENT
R(s) for the same (concept, source form, document). Only the glossary-backed
references constitute a common yardstick.

Each reference carries TWO forms:

- ``reference``       : the natural/display form: shown to the LLM (Q1/Q2),
                        displayed in the chains, stored in the annotation.
- ``reference_lemma`` : the lemma form: used ONLY for the divergence match,
                        so number/case do not create false divergences.

Symmetry with the target: the match compares ``target_lemma`` to
``reference_lemma``; the LLM always sees the raw / natural forms.

Strategies (combine via fallback chain or presets):
- ``glossary``   : prefLabel_fr of the concept, ONLY for the head term; its
                   lemma comes from the glossary FR-lemma sidecar.
- ``frequency``  : the most frequent target LEMMA in the doc (a natural
                   representative is kept for display).
- ``first``      : the first observed target.
- ``llm``        : LLM picks among the candidates; SHORT-CIRCUITS if a single
                   distinct candidate (no API call).
- ``generate``   : LLM PRODUCES the canonical French form for this source form
                   from the concept (head term) + SOURCE-side contexts only,
                   never the systems' outputs (anti-circularity; the prompt and
                   its cache entry are shared by all systems). Retries once at
                   a higher temperature on failure; falls through on decline.
                   Gated by ``--generate-scope`` (all|variants|derived|missing).
- ``validated``  : meta-strategy. Pick the heuristic reference
                   (``glossary > frequency > first``), then VALIDATE it with
                   the error judge: if the heuristic reference is itself a
                   terminology ERROR (e.g. a circular frequency reference = the
                   system's own mistake), GENERATE a fresh reference instead;
                   if it is valid (judge = NO), keep it. The glossary head term
                   is trusted without validation. Keeps a cheap correct reference
                   when it is right and only breaks circularity where it is wrong.

Presets:
- ``glossary``              : ``glossary > frequency > first``
- ``glossary_only_or_skip`` : ``glossary``  (no fallback)
- ``frequency``             : ``frequency > first``
- ``first``                 : ``first``
- ``llm``                   : ``llm > glossary > frequency > first``
- ``generate``              : ``glossary > generate > frequency > first``
- ``validated``             : ``validated``  (glossary>frequency>first, judge-gated -> generate)

Or pass an explicit chain: ``glossary>generate>frequency>first``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..io.skos import GlossaryEntry
from ..llm.client import LLMClient
from .variation import develop as develop_variation


_STRATEGY_PREFIX_RE = re.compile(r"^(?:glossary|frequency|first)\s*:\s*", re.IGNORECASE)


def _norm_choice(text: str) -> str:
    """Normalise an LLM-returned reference for robust matching against the
    displayed candidate forms.

    The candidates are shown to the model wrapped in quotes and prefixed by a
    strategy name; a model may copy the quotes, "correct" a typographic
    apostrophe, add a stray space, or keep the strategy prefix. Matching the
    raw string against the bare natural forms then fails silently and the
    ``llm`` strategy degrades to the next one with no trace. This normaliser
    absorbs exactly those cosmetic differences so the comparison stays robust.

    It is deliberately LIGHT (no lowercasing, no diacritics folding) so that
    two genuinely distinct candidate forms never collapse onto the same key.
    """
    if not text:
        return ""
    s = str(text).strip()
    s = _STRATEGY_PREFIX_RE.sub("", s).strip()
    # Drop one layer of surrounding quotes the model may have copied from the
    # display (straight single/double, or French guillemets).
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    elif len(s) >= 2 and s[0] == "«" and s[-1] == "»":  # « »
        s = s[1:-1].strip()
    # Unify apostrophe variants (typographic vs straight) and collapse spaces.
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------------------------------------------------------------------
# Public datatypes
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    """One observed occurrence of a (concept, src_form) across systems/docs."""

    system: str
    doc_id: str
    src_sent_ids: Tuple[int, ...]
    target_canonical: str          # natural form (for display / representative)
    target_lemma: str = ""         # lemma (for counting / matching)
    target_raw: str = ""           # raw aligned surface form (grammatical; for validation)


@dataclass
class ReferenceChoice:
    reference: str                 # natural form -> LLM + display + annotation
    reference_lemma: str           # lemma -> match_level only
    source: str                    # which strategy produced this reference
    confidence: str                # "high" | "medium" | "low"
    candidates: Dict[str, str]     # natural candidates (shown to the LLM in select_reference)
    observed_distribution: List[Tuple[str, int]]
    justification: str = ""
    llm_short_circuited: bool = False
    generation_trace: Optional[dict] = None  # audit of the "generate" strategy (contexts shown, attempts, status)
    ref_raw: str = ""              # raw surface form of the representative occurrence (for validation)


PRESETS: Dict[str, List[str]] = {
    "glossary": ["glossary", "frequency", "first"],
    "glossary_only_or_skip": ["glossary"],
    "frequency": ["frequency", "first"],
    "first": ["first"],
    "llm": ["llm", "glossary", "frequency", "first"],
    "generate": ["glossary", "generate", "frequency", "first"],
    "validated": ["validated"],
}


def parse_mode(mode: str) -> List[str]:
    """Resolve a CLI ``--reference-mode`` value to a strategy chain."""
    mode = (mode or "").strip()
    if not mode:
        return PRESETS["glossary"]
    if mode in PRESETS:
        return PRESETS[mode]
    chain = [s.strip() for s in mode.split(">") if s.strip()]
    if not chain:
        raise ValueError(f"Empty reference-mode chain: {mode!r}")
    valid = {"glossary", "frequency", "first", "llm", "generate", "validated"}
    bad = [s for s in chain if s not in valid]
    if bad:
        raise ValueError(f"Unknown strategies in reference-mode {mode!r}: {bad}")
    return chain


# ---------------------------------------------------------------------------
# Candidate computation
# ---------------------------------------------------------------------------


@dataclass
class _Candidates:
    natural: Dict[str, str] = field(default_factory=dict)   # strategy -> natural form
    lemma: Dict[str, str] = field(default_factory=dict)     # strategy -> lemma form
    distribution: List[Tuple[str, int]] = field(default_factory=list)  # (lemma, count)
    natural_of: Dict[str, str] = field(default_factory=dict)  # lemma -> natural representative
    raw_of: Dict[str, str] = field(default_factory=dict)     # lemma -> raw surface representative


def _candidates(
    observations: List[Observation],
    glossary_entry: Optional[GlossaryEntry],
    is_vedette: bool = False,
) -> _Candidates:
    ordered = sorted(observations, key=lambda o: (o.doc_id, o.src_sent_ids, o.system))

    # Count on the LEMMA so that singular/plural/case variants of the same
    # target group together; keep a natural representative per lemma for display.
    counts: Counter = Counter()
    natural_of: Dict[str, str] = {}
    raw_of: Dict[str, str] = {}
    first_seen: Dict[str, int] = {}
    for idx, o in enumerate(ordered):
        lem = o.target_lemma
        if not lem:
            continue
        counts[lem] += 1
        if lem not in natural_of:
            natural_of[lem] = o.target_canonical or lem
            # raw surface form of the SAME representative occurrence (grammatical,
            # used to validate the reference: the canonical form can be a mangled
            # lemma-like string for verb phrases, e.g. "être effectuer").
            raw_of[lem] = o.target_raw or o.target_canonical or lem
            first_seen[lem] = idx

    distribution = sorted(
        counts.items(),
        key=lambda kv: (-kv[1], first_seen.get(kv[0], 10**9)),
    )

    cand = _Candidates(distribution=distribution, natural_of=natural_of, raw_of=raw_of)

    # Glossary gives a usable FR reference ONLY for the head term: prefLabel_fr
    # is the head term's translation, not a variant's. For a variant source
    # form we leave glossary out and fall back to frequency/first (the
    # document's own established rendering of THAT variant). The reference
    # lemma is the sidecar lemma; if the sidecar is missing it falls back to
    # the raw prefLabel_fr (a warning is emitted at glossary load).
    if is_vedette and glossary_entry and glossary_entry.preflabel_fr:
        cand.natural["glossary"] = glossary_entry.preflabel_fr
        cand.lemma["glossary"] = glossary_entry.preflabel_fr_lemma or glossary_entry.preflabel_fr

    if distribution:
        top_lemma = distribution[0][0]
        cand.natural["frequency"] = natural_of[top_lemma]
        cand.lemma["frequency"] = top_lemma

    for o in ordered:
        if o.target_lemma:
            cand.natural["first"] = o.target_canonical or o.target_lemma
            cand.lemma["first"] = o.target_lemma
            break

    return cand


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _make(cand: _Candidates, strat: str, confidence: str, **kw) -> ReferenceChoice:
    return ReferenceChoice(
        reference=cand.natural[strat],
        reference_lemma=cand.lemma[strat],
        source=strat,
        confidence=confidence,
        candidates=cand.natural,
        observed_distribution=cand.distribution,
        ref_raw=cand.raw_of.get(cand.lemma[strat], cand.natural[strat]),
        **kw,
    )


def _try_glossary(cand: _Candidates) -> Optional[ReferenceChoice]:
    if "glossary" in cand.natural:
        return _make(cand, "glossary", "high")
    return None


def _try_frequency(cand: _Candidates) -> Optional[ReferenceChoice]:
    if "frequency" not in cand.natural:
        return None
    max_count = cand.distribution[0][1] if cand.distribution else 0
    return _make(cand, "frequency", "medium" if max_count >= 2 else "low")


def _try_first(cand: _Candidates) -> Optional[ReferenceChoice]:
    if "first" in cand.natural:
        return _make(cand, "first", "low")
    return None


def _try_llm(
    cand: _Candidates,
    *,
    llm: Optional[LLMClient],
    prompt_tmpl: Optional[str],
    src_form: str,
    preflabel_en: str,
    doc_id: str,
) -> Optional[ReferenceChoice]:
    """Ask the LLM to pick among the DISTINCT candidates (natural forms).

    Distinctness is on the lemma (singular/plural variants already grouped), so
    each distinct choice is one lemma, shown via a natural representative. The
    LLM never sees a lemma; its chosen natural form is mapped back to its lemma.
    Short-circuits (no API call) when a single distinct candidate remains.
    """
    # natural -> lemma over candidates + observed representatives
    nat2lem: Dict[str, str] = {}
    for strat, nat in cand.natural.items():
        nat2lem[nat] = cand.lemma[strat]
    for lem, _ in cand.distribution:
        nat2lem.setdefault(cand.natural_of.get(lem, lem), lem)

    distinct = [n for n in nat2lem if n]
    if not distinct:
        return None

    if len(distinct) == 1:
        chosen = distinct[0]
        return ReferenceChoice(
            reference=chosen,
            reference_lemma=nat2lem[chosen],
            source="llm_short_circuit",
            confidence="high",
            candidates=cand.natural,
            observed_distribution=cand.distribution,
            justification="Single distinct candidate; LLM call skipped.",
            llm_short_circuited=True,
        )

    if llm is None or prompt_tmpl is None:
        return None

    # Display forms in CONSISTENT double quotes; the matching below is
    # quote-insensitive regardless.
    observed_text = "\n".join(
        f'  - "{cand.natural_of.get(lem, lem)}" × {n}' for lem, n in cand.distribution[:20]
    )
    cand_text = "\n".join(f'  - {k}: "{v}"' for k, v in cand.natural.items())
    user = prompt_tmpl.format(
        src_form=src_form,
        preflabel_en=preflabel_en or "(unknown)",
        doc_id=doc_id,
        candidates=cand_text or "  (none)",
        distribution=observed_text or "  (none)",
    )
    parsed = llm.chat_json(
        prompt_name="select_reference",
        system="You are a bilingual (English-French) terminologist selecting the best reference translation for a source term in a document.",
        user=user,
        dump_label=f"{doc_id}__{src_form}",
    )
    chosen_raw = None
    justif = ""
    if isinstance(parsed, dict):
        chosen_raw = parsed.get("reference")
        justif = parsed.get("justification", "") or ""
    # Robust match: compare the model's answer to the candidate natural forms
    # after normalisation (quotes, strategy prefix, apostrophe variant, spacing),
    # then map back to the EXACT natural form so downstream matching stays exact.
    chosen = None
    if isinstance(chosen_raw, str) and chosen_raw:
        if chosen_raw in nat2lem:            # fast path: exact copy
            chosen = chosen_raw
        else:
            norm_to_nat: Dict[str, str] = {}
            for nat in nat2lem:
                norm_to_nat.setdefault(_norm_choice(nat), nat)
            chosen = norm_to_nat.get(_norm_choice(chosen_raw))
    if not chosen:
        return None
    return ReferenceChoice(
        reference=chosen,
        reference_lemma=nat2lem[chosen],
        source="llm",
        confidence="high",
        candidates=cand.natural,
        observed_distribution=cand.distribution,
        justification=justif,
    )


# ---------------------------------------------------------------------------
# "generate" strategy: PRODUCE the reference from the concept (LLM knowledge),
# instead of selecting among the system's own outputs. This breaks the
# circularity of frequency/first for variant forms (the system's error would
# otherwise become its own gold standard).
# ---------------------------------------------------------------------------


GENERATE_SCOPES = ("all", "variants", "derived", "missing")

# Variant categories DERIVABLE from the head term (vs VL = independent synonym)
_DERIVED_CATEGORIES = {"vg", "vr", "ve", "vms", "cm"}

_GEN_SYSTEM_MSG = (
    "You are a bilingual English-French terminologist. Given a concept and one "
    "source form that designates it, you produce the canonical French form "
    "expected for that source form, or decline when no established form exists."
)

_GEN_PLACEHOLDER_RE = re.compile(r"^\(.*\)$")


def _generate_eligible(scope: str, is_vedette: bool, variation_category: str,
                       cand: _Candidates) -> bool:
    """Gate the generate strategy by form category (``--generate-scope``)."""
    if scope == "all":
        return True
    if scope == "variants":
        return not is_vedette
    if scope == "derived":
        return (not is_vedette) and \
            (variation_category or "").strip().lower() in _DERIVED_CATEGORIES
    if scope == "missing":
        return "glossary" not in cand.natural
    return False


def _select_contexts(contexts: List[Tuple[object, str]], n: int) -> List[Tuple[object, str]]:
    """Deterministic even-spread pick of ``n`` (seg_id, sentence) items.

    First / evenly spaced middle / last occurrence: covers the (often defining)
    first mention plus running usage, is fully positional (no embeddings, no
    randomness) so the prompt, and its cache key, are reproducible.
    """
    if n <= 0 or not contexts:
        return []
    if len(contexts) <= n:
        return list(contexts)
    if n == 1:
        return [contexts[0]]
    idxs = [round(i * (len(contexts) - 1) / (n - 1)) for i in range(n)]
    seen, out = set(), []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(contexts[i])
    return out


def _truncate_around(sentence: str, form: str, width: int = 300) -> str:
    """Bound a context sentence, keeping a window centred on the form."""
    if len(sentence) <= width:
        return sentence
    low, f = sentence.lower(), (form or "").lower()
    pos = low.find(f) if f else -1
    if pos < 0:
        return sentence[:width] + "…"
    start = max(0, pos - max(0, (width - len(f)) // 2))
    end = min(len(sentence), start + width)
    return ("…" if start > 0 else "") + sentence[start:end] + ("…" if end < len(sentence) else "")


def _validate_generation(parsed) -> Tuple[str, Optional[str], str, str]:
    """Return (status, reference, confidence, justification) for one attempt."""
    if not isinstance(parsed, dict):
        return "parse_failed", None, "medium", ""
    justif = str(parsed.get("analysis") or parsed.get("justification") or "")
    if parsed.get("generatable") is False:
        return "declined", None, "medium", justif
    ref = parsed.get("reference")
    if not isinstance(ref, str):
        return "invalid", None, "medium", justif
    ref = " ".join(ref.split()).strip()
    if not ref or len(ref) > 200 or "\n" in ref or _GEN_PLACEHOLDER_RE.match(ref):
        return "invalid", None, "medium", justif
    conf = str(parsed.get("confidence") or "medium").strip().lower()
    if conf not in ("high", "medium", "low"):
        conf = "medium"
    return "ok", ref, conf, justif


def _try_generate(
    cand: _Candidates,
    *,
    llm: Optional[LLMClient],
    prompt_tmpl: Optional[str],
    src_form: str,
    variation_category: str,
    preflabel_en: str,
    preflabel_fr: str,
    doc_id: str,
    contexts: List[Tuple[object, str]],
    scope: str,
    n_contexts: int,
    retry_temperature: Optional[float],
    is_vedette: bool,
) -> Tuple[Optional[ReferenceChoice], Optional[dict]]:
    """Generate the canonical French reference for this source form.

    Anti-circularity by construction: the prompt uses ONLY concept/glossary
    information and SOURCE-side contexts (identical for every MT system), never
    the systems' translations. The same prompt therefore hits the same cache
    entry for all systems (one API call shared by the whole comparison).

    Attempt 1 runs at the client's default temperature (0.0); if it fails
    (decline / unparseable / invalid), one retry runs at ``retry_temperature``
    (None disables the retry). Returns ``(choice_or_None, trace_or_None)``;
    the trace is returned even on failure so the caller can persist it.
    """
    if llm is None or prompt_tmpl is None:
        return None, None
    if not _generate_eligible(scope, is_vedette, variation_category, cand):
        return None, None

    picked = _select_contexts(contexts or [], n_contexts)
    if picked:
        ctx_text = "\n".join(
            f"  - [seg {sid}] {_truncate_around(txt, src_form)}" for sid, txt in picked
        )
    else:
        ctx_text = "  (none)"
    user = prompt_tmpl.format(
        src_form=src_form or "(empty)",
        variation_category=develop_variation(variation_category),
        preflabel_en=preflabel_en or "(not in glossary)",
        preflabel_fr=preflabel_fr or "(not in glossary)",
        doc_id=doc_id,
        contexts=ctx_text,
    )

    temps: List[Optional[float]] = [None]  # None -> client default (0.0)
    if retry_temperature is not None:
        temps.append(retry_temperature)

    attempts = []
    result = None
    for t in temps:
        parsed = llm.chat_json(
            prompt_name="generate_reference",
            system=_GEN_SYSTEM_MSG,
            user=user,
            temperature=t,
            dump_label=f"{doc_id}__{src_form}",
        )
        status, ref, conf, justif = _validate_generation(parsed)
        attempts.append({
            "temperature": "default" if t is None else t,
            "status": status,
            "reference": ref,
        })
        if status == "ok":
            result = (ref, conf, justif)
            break
        # decline / parse_failed / invalid at this temperature: try the next
        # (higher) temperature before giving up. A decline at T=0 may become a
        # valid reference at a higher temperature; falling straight back to
        # frequency would reintroduce the circular pseudo-reference.

    trace = {
        "doc_id": doc_id,
        "src_form": src_form,
        "variation_category": variation_category or "",
        "preflabel_en": preflabel_en or "",
        "preflabel_fr": preflabel_fr or "",
        "n_occurrences": len(contexts or []),
        "context_selection": f"even_spread_n{n_contexts}" if picked else "none",
        "contexts_shown": [
            {"seg": sid, "text": _truncate_around(txt, src_form)} for sid, txt in picked
        ],
        "attempts": attempts,
        "status": "generated" if result else (
            "declined" if any(a["status"] == "declined" for a in attempts) else "failed"
        ),
    }
    if result is None:
        return None, trace
    ref, conf, justif = result
    trace["reference"] = ref
    trace["confidence"] = conf
    trace["justification"] = justif
    choice = ReferenceChoice(
        reference=ref,
        # No lemma for a generated form: fall back to the form itself (same
        # mechanism as a glossary entry without its lemma sidecar).
        reference_lemma=ref,
        source="generate",
        confidence=conf,
        candidates=cand.natural,
        observed_distribution=cand.distribution,
        justification=justif,
        generation_trace=trace,
    )
    return choice, trace


# ---------------------------------------------------------------------------
# "validated" meta-strategy: pick the heuristic reference, validate it with
# the error judge, and GENERATE a replacement only when the heuristic
# reference is itself a terminology error (breaks circularity where it is
# wrong, keeps the cheap correct reference where it is right).
# ---------------------------------------------------------------------------


def _validate_reference(
    llm: LLMClient,
    hapax_prompt_tmpl: str,
    *,
    src_form: str,
    candidate: str,
    doc_id: str,
    preflabel_en: str,
    preflabel_fr: str,
    variation_category: str,
    contexts: List[Tuple[object, str]],
    n_contexts: int,
    system: Optional[str] = None,
    chain_label: Optional[str] = None,
    prompt_name: str = "hapax_judge",
    evidence: Optional[dict] = None,
) -> Tuple[str, str]:
    """Error judge on a CANDIDATE reference: is it a valid French translation
    of the source form, judged in the absolute (no meta-reference)?

    Returns ``(code, justification)``; ``code == "NO"`` means the candidate is
    acceptable terminology.
    """
    from .hapax import judge as hapax_judge

    # IN-CONTEXT validation: the candidate form comes from a specific
    # occurrence, so it is judged as Q1 judges any alignment: its source
    # segment, its translation and the same supporting blocks. Otherwise, the
    # source-only construction (byte for byte, cache preserved).
    if evidence:
        res = hapax_judge(
            llm=llm,
            prompt_tmpl=hapax_prompt_tmpl,
            src_segment=evidence.get("src_segment", ""),
            tgt_segment=evidence.get("tgt_segment", ""),
            src_form_raw=evidence.get("src_form_raw") or src_form,
            tgt_aligned_raw=evidence.get("tgt_aligned_raw") or candidate,
            src_canonical=src_form,
            tgt_canonical=candidate,
            concept_preflabel_en=preflabel_en,
            concept_preflabel_fr=preflabel_fr,
            variation_category=variation_category,
            concept_chain="",
            reference=None,
            context_window="",
            context_pairs=evidence.get("context_pairs", ""),
            concept_history_indexed=evidence.get("concept_history_indexed", ""),
            system=system,
            prompt_name=prompt_name,
            dump_name="validate_reference",
            dump_label=f"{doc_id}__{src_form}__{candidate}",
        )
        return res.code, res.justification

    picked = _select_contexts(contexts or [], n_contexts)
    src_seg = picked[0][1] if picked else ""
    # ``src_seg`` is ALREADY the first context: when the section header is
    # explicit, the chain only lists the OTHER segments, otherwise the same
    # segment would appear twice (raw then truncated). Without an overridden
    # header, the historical construction is kept byte for byte.
    rest = picked[1:] if chain_label else picked
    chain_txt = "\n".join(
        f"  - [seg {sid}] {_truncate_around(txt, src_form)}" for sid, txt in rest
    ) or "(no other in-document context)"
    res = hapax_judge(
        llm=llm,
        prompt_tmpl=hapax_prompt_tmpl,
        src_segment=src_seg,
        tgt_segment="",
        src_form_raw=src_form,
        tgt_aligned_raw=candidate,
        src_canonical=src_form,
        tgt_canonical=candidate,
        concept_preflabel_en=preflabel_en,
        concept_preflabel_fr=preflabel_fr,
        variation_category=variation_category,
        concept_chain=chain_txt,
        reference=None,   # judge the candidate itself, no expected form
        context_window="",
        # Reuses the hapax_judge PROMPT (hence the SAME cache), but SAVES these
        # calls in a distinct ``validate_reference/`` folder with a label
        # identifying the validated form. dump_name does NOT touch the cache key.
        dump_name="validate_reference",
        dump_label=f"{doc_id}__{src_form}__{candidate}",
        # None => default system message (cache key preserved).
        system=system,
        chain_label=chain_label,
    )
    return res.code, res.justification


def _try_validated(
    cand: _Candidates,
    *,
    llm: Optional[LLMClient],
    hapax_prompt_tmpl: Optional[str],
    generate_prompt_tmpl: Optional[str],
    src_form: str,
    variation_category: str,
    preflabel_en: str,
    preflabel_fr: str,
    doc_id: str,
    contexts: List[Tuple[object, str]],
    n_contexts: int,
    retry_temperature: Optional[float],
    is_vedette: bool,
    validate_glossary: bool = False,
    hapax_validate_system: Optional[str] = None,
    hapax_validate_chain_label: Optional[str] = None,
    hapax_validate_prompt_name: str = "hapax_judge",
    validation_evidence: Optional[Callable[[str], Optional[dict]]] = None,
) -> Tuple[Optional[ReferenceChoice], Optional[dict]]:
    """glossary > frequency > first, judge-gated, generate-on-error."""
    heuristic = _try_glossary(cand) or _try_frequency(cand) or _try_first(cand)

    def _generate(status_reason: str, vtrace: Optional[dict]):
        # Force eligibility: the decision to generate is made by the validation,
        # not by --generate-scope (we already know the heuristic ref is wrong).
        gen_choice, gen_trace = _try_generate(
            cand, llm=llm, prompt_tmpl=generate_prompt_tmpl, src_form=src_form,
            variation_category=variation_category, preflabel_en=preflabel_en,
            preflabel_fr=preflabel_fr, doc_id=doc_id, contexts=contexts,
            scope="all", n_contexts=n_contexts, retry_temperature=retry_temperature,
            is_vedette=is_vedette,
        )
        if gen_choice is not None:
            tr = dict(gen_choice.generation_trace or {}, status=status_reason)
            if vtrace is not None:
                tr["validation"] = vtrace
            gen_choice.generation_trace = tr
            return gen_choice, tr
        return None, gen_trace

    if heuristic is None:
        # nothing observed to validate -> generate directly
        return _generate("generated_no_heuristic", None)

    # glossary head term is authoritative -> trust it without validation
    if heuristic.source == "glossary" and not validate_glossary:
        return heuristic, None

    # cannot validate (no LLM / prompt) -> keep the heuristic as-is
    if llm is None or hapax_prompt_tmpl is None:
        return heuristic, None

    # Validate the RAW surface form (grammatical), not the canonical form which
    # can be a mangled lemma-like string for verb phrases ("être effectuer").
    candidate_shown = heuristic.ref_raw or heuristic.reference
    code, justif = _validate_reference(
        llm, hapax_prompt_tmpl, src_form=src_form, candidate=candidate_shown,
        doc_id=doc_id,
        preflabel_en=preflabel_en, preflabel_fr=preflabel_fr,
        variation_category=variation_category, contexts=contexts, n_contexts=n_contexts,
        system=hapax_validate_system,
        chain_label=hapax_validate_chain_label,
        prompt_name=hapax_validate_prompt_name,
        evidence=(validation_evidence(heuristic.reference_lemma)
                  if validation_evidence is not None else None),
    )
    vtrace = {
        "src_form": src_form,
        "doc_id": doc_id,
        "candidate": heuristic.reference,
        "candidate_validated": candidate_shown,
        "candidate_source": heuristic.source,
        "q1bis_code": code,
        "q1bis_justification": justif,
    }

    if code == "NO":   # heuristic reference is valid -> keep it
        heuristic.confidence = "high"
        heuristic.justification = f"Q1_bis validated the {heuristic.source} reference (NO). {justif}".strip()
        heuristic.generation_trace = {"validation": vtrace, "status": "validated_heuristic"}
        return heuristic, heuristic.generation_trace

    # heuristic reference is itself an error -> generate a replacement
    gen_choice, gen_trace = _generate("generated_after_invalid_heuristic", vtrace)
    if gen_choice is not None:
        gen_choice.justification = (
            f"Q1_bis rejected the {heuristic.source} reference ({code}); generated a replacement. "
            + (gen_choice.justification or "")
        ).strip()
        return gen_choice, gen_choice.generation_trace

    # generation failed -> keep the (imperfect) heuristic so a reference still exists
    heuristic.justification = (
        f"Q1_bis rejected the {heuristic.source} reference ({code}) but generation failed; kept heuristic."
    )
    heuristic.generation_trace = dict((gen_trace or {}), validation=vtrace,
                                      status="generate_failed_kept_heuristic")
    return heuristic, heuristic.generation_trace


# ---------------------------------------------------------------------------
# Public selector
# ---------------------------------------------------------------------------


def select_reference(
    chain: Sequence[str],
    src_form: str,
    glossary_entry: Optional[GlossaryEntry],
    doc_id: str,
    observations: List[Observation],
    llm: Optional[LLMClient] = None,
    llm_prompt_tmpl: Optional[str] = None,
    is_vedette: bool = False,
    generate_prompt_tmpl: Optional[str] = None,
    variation_category: str = "",
    source_contexts: Optional[List[Tuple[object, str]]] = None,
    generate_scope: str = "variants",
    generate_contexts: int = 3,
    generate_retry_temperature: Optional[float] = 0.7,
    hapax_prompt_tmpl: Optional[str] = None,
    hapax_validate_system: Optional[str] = None,
    hapax_validate_chain_label: Optional[str] = None,
    hapax_validate_prompt_name: str = "hapax_judge",
    validation_evidence: Optional[Callable[[str], Optional[dict]]] = None,
    validate_glossary: bool = False,
) -> ReferenceChoice:
    """Pick R(s) using a strategy chain. First strategy that yields a non-empty reference wins.

    ``is_vedette`` gates the glossary strategy: the glossary prefLabel_fr is only
    a valid reference for the head term, not for variant source forms.

    The ``generate`` strategy (see :func:`_try_generate`) PRODUCES the reference
    from the concept + SOURCE-side contexts; on failure it falls through to the
    next strategy, and its trace is attached to the winning choice
    (``generation_trace`` with ``fallback_source``) so failures stay auditable.
    """
    cand = _candidates(observations, glossary_entry, is_vedette)
    preflabel_en = glossary_entry.preflabel_en if glossary_entry else ""
    preflabel_fr = glossary_entry.preflabel_fr if glossary_entry else ""

    gen_trace: Optional[dict] = None
    for strat in chain:
        if strat == "glossary":
            res = _try_glossary(cand)
        elif strat == "frequency":
            res = _try_frequency(cand)
        elif strat == "first":
            res = _try_first(cand)
        elif strat == "generate":
            res, trace = _try_generate(
                cand,
                llm=llm,
                prompt_tmpl=generate_prompt_tmpl,
                src_form=src_form,
                variation_category=variation_category,
                preflabel_en=preflabel_en,
                preflabel_fr=preflabel_fr,
                doc_id=doc_id,
                contexts=source_contexts or [],
                scope=generate_scope,
                n_contexts=generate_contexts,
                retry_temperature=generate_retry_temperature,
                is_vedette=is_vedette,
            )
            if trace is not None:
                gen_trace = trace
        elif strat == "validated":
            res, trace = _try_validated(
                cand,
                llm=llm,
                hapax_prompt_tmpl=hapax_prompt_tmpl,
                generate_prompt_tmpl=generate_prompt_tmpl,
                src_form=src_form,
                variation_category=variation_category,
                preflabel_en=preflabel_en,
                preflabel_fr=preflabel_fr,
                doc_id=doc_id,
                contexts=source_contexts or [],
                n_contexts=generate_contexts,
                retry_temperature=generate_retry_temperature,
                is_vedette=is_vedette,
                validate_glossary=validate_glossary,
                hapax_validate_system=hapax_validate_system,
                hapax_validate_chain_label=hapax_validate_chain_label,
                hapax_validate_prompt_name=hapax_validate_prompt_name,
                validation_evidence=validation_evidence,
            )
            if trace is not None:
                gen_trace = trace
        elif strat == "llm":
            res = _try_llm(
                cand,
                llm=llm,
                prompt_tmpl=llm_prompt_tmpl,
                src_form=src_form,
                preflabel_en=preflabel_en,
                doc_id=doc_id,
            )
        else:
            raise ValueError(f"Unknown strategy: {strat}")
        if res is not None and res.reference:
            if gen_trace is not None and res.generation_trace is None:
                # generation failed and a later strategy won: keep the audit trail
                res.generation_trace = dict(gen_trace, fallback_source=res.source)
            return res

    final = ReferenceChoice(
        reference="",
        reference_lemma="",
        source="none",
        confidence="low",
        candidates=cand.natural,
        observed_distribution=cand.distribution,
        justification=f"No strategy in chain {list(chain)} yielded a reference.",
    )
    if gen_trace is not None:
        final.generation_trace = dict(gen_trace, fallback_source="none")
    return final


# ---------------------------------------------------------------------------
# Observations collection
# ---------------------------------------------------------------------------


def collect_observations(
    records_by_system: Dict[str, List[dict]],
    source_key: callable,
    canonical_tgt: callable,
    target_lemma: callable,
    concept_key: callable,
    raw_target: callable,
) -> Dict[Tuple[str, str], List[Observation]]:
    """Group occurrences by ``(concept, normalised source LEMMA)``.

    Grouping on the source lemma (``source_key``) means singular/plural/case
    variants of the same source term are not split into separate forms. Each
    Observation carries the natural target (``canonical_tgt``, for display),
    the target lemma (``target_lemma``, for counting/matching) and the raw
    aligned span (``raw_target``, for validation).
    """
    from .match import normalise

    out: Dict[Tuple[str, str], List[Observation]] = {}
    for system, records in records_by_system.items():
        for r in records:
            key_concept = concept_key(r)
            if not key_concept:
                continue
            src = source_key(r)
            if not src:
                continue
            key = (key_concept, normalise(src))
            out.setdefault(key, []).append(
                Observation(
                    system=system,
                    doc_id=r.get("doc_id", ""),
                    src_sent_ids=tuple(r.get("source_segment_ids") or ()),
                    target_canonical=canonical_tgt(r),
                    target_lemma=target_lemma(r),
                    target_raw=(raw_target(r) or canonical_tgt(r)),
                )
            )
    return out

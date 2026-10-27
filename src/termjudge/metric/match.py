"""Compare a target phrase to the reference R(s) on canonical forms.

We assume canonical forms are already produced upstream
(``tgt_phrase_final_form_canonical`` for the target,
``variation_words`` for the source). Singular/plural and inflection are
expected to be normalised by that upstream step. Matching here is light:

- ``exact``    : strings equal after lowercase + whitespace + edge punctuation
                 + leading French articles (``le``, ``la``, ``les``, ``l'`` ...)
- ``graphic``  : equal after additionally collapsing dashes/apostrophes and
                 stripping diacritics - used by the Q2 equivalence rule
- ``none``     : no match → divergence, route to Q1
"""

from __future__ import annotations

import re
import unicodedata


WHITESPACE_RE = re.compile(r"\s+")
STRIP_EDGES_RE = re.compile(r"^[\s\.,;:!\?\(\)\[\]\"\'«»]+|[\s\.,;:!\?\(\)\[\]\"\'«»]+$")
LEADING_ARTICLES = (
    "l'", "d'", "le ", "la ", "les ", "un ", "une ", "des ", "de ",
    "du ", "au ", "aux ",
)


def normalise(text: str) -> str:
    if not text:
        return ""
    s = text.lower()
    s = STRIP_EDGES_RE.sub("", s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    for art in LEADING_ARTICLES:
        if s.startswith(art):
            s = s[len(art):].strip()
            break
    return s


def _strip_diacritics(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalise_graphic(text: str) -> str:
    s = normalise(text)
    s = s.replace("-", " ").replace("'", " ").replace("’", " ")
    s = WHITESPACE_RE.sub(" ", s).strip()
    s = _strip_diacritics(s)
    return s


def match_level(target: str, reference: str) -> str:
    if not target or not reference:
        return "none"
    if normalise(target) == normalise(reference):
        return "exact"
    if normalise_graphic(target) == normalise_graphic(reference):
        return "graphic"
    return "none"


def any_equivalence(target: str, candidates) -> bool:
    """True if ``target`` matches any candidate at graphic level."""
    t_norm = normalise_graphic(target)
    if not t_norm:
        return False
    for cand in candidates:
        if cand and normalise_graphic(cand) == t_norm:
            return True
    return False

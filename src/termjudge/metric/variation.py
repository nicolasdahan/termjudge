"""Human-readable rendering of the source variation category.

Occurrence files may carry a ``variation_category`` code describing how the
source form relates to the head term of its concept (``no_variation``,
``VG``, ``VR``, ``VE``, ``VL``, ``VMS``, ``CM``; see ``docs/input_format.md``).
The codes are opaque to the LLM, so every prompt-facing call site renders
them through :func:`develop` and the judge reads the developed name
(e.g. "graphical variation") instead of the bare code. Composite categories
(``CM``, or any ``X+Y`` combination) collapse to "composite form". Outputs
and eligibility checks keep the raw code; this rendering is for prompts only.
"""

from __future__ import annotations

# Bare code (lowercased) -> developed name shown to the LLM.
_DEVELOPED = {
    "no_variation": "none (the source form is the head term itself)",
    "vg": "graphical variation",
    "vr": "reduction",
    "ve": "expansion",
    "vl": "lexical variation",
    "vms": "morphosyntactic variation",
    "cm": "composite form",
}

_UNKNOWN = {"", "(none)", "none", "unknown", "(unknown)"}


def develop(raw: str) -> str:
    """Return the developed name for a variation category code.

    - Known single code -> its developed name.
    - Composite (``CM`` or any ``X+Y``) -> "composite form".
    - Empty / unknown placeholder -> "(unknown)".
    - Anything unrecognised is returned unchanged (fail-open).
    """
    s = (raw or "").strip()
    if s.lower() in _UNKNOWN:
        return "(unknown)"
    if "+" in s:  # e.g. "VG+VL": a combination, never spelled out one by one
        return "composite form"
    return _DEVELOPED.get(s.lower(), s)

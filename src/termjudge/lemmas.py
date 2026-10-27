"""Lemmatise the French side of the inputs with spaCy (optional extra ``[lemmas]``).

TermJudge detects a divergence by comparing the lemma of the aligned target
span with the lemma of the reference, so that number and case do not create
false divergences. Both lemmas must come from the same lemmatiser. This
module provides that lemmatiser for French, with the spaCy model used in the
paper (``fr_core_news_lg``), for the two inputs that need it:

- the occurrence files, where it fills the optional ``target_lemma`` field
  from ``target_canonical`` (or ``target_term``);
- the glossary, where it writes the sidecar ``<glossary_stem>.fr_lemmas.json``
  mapping each ``skos:prefLabel@fr`` to its lemma.

Install the extra and the model first::

    pip install "termjudge[lemmas]"
    python -m spacy download fr_core_news_lg

Then::

    python -m termjudge.lemmas occurrences --input-dir alignments/ [--output-dir out/]
    python -m termjudge.lemmas glossary --glossary glossary.csv [--output glossary.fr_lemmas.json]

Without the extra, TermJudge still runs: the match falls back to the
canonical or raw forms (see ``docs/input_format.md``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from .io.records import write_doc

DEFAULT_MODEL = "fr_core_news_lg"


def load_model(name: str = DEFAULT_MODEL):
    try:
        import spacy
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "spaCy is not installed. Install the optional extra with: "
            "pip install \"termjudge[lemmas]\""
        ) from e
    try:
        return spacy.load(name)
    except OSError as e:  # pragma: no cover
        raise SystemExit(
            f"spaCy model {name!r} not found. Download it with: "
            f"python -m spacy download {name}"
        ) from e


def lemmatise(nlp, text: str) -> str:
    """Space-joined token lemmas of ``text`` (empty for an empty text)."""
    if not text or not text.strip():
        return ""
    doc = nlp(text.strip())
    return " ".join(tok.lemma_ for tok in doc if not tok.is_space).strip()


def lemmatise_records(records: Iterable[dict], nlp, overwrite: bool = False) -> int:
    """Fill ``target_lemma`` in place for each occurrence. Returns the number of
    records updated. Records with an empty target keep no lemma."""
    n = 0
    for r in records:
        if r.get("target_lemma") and not overwrite:
            continue
        text = r.get("target_canonical") or r.get("target_term") or ""
        lemma = lemmatise(nlp, text)
        if lemma:
            r["target_lemma"] = lemma
            n += 1
    return n


def _run_occurrences(args: argparse.Namespace) -> None:
    nlp = load_model(args.spacy_model)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    files = sorted(input_dir.glob("*/*.json"))
    if not files:
        raise SystemExit(f"No <SYSTEM>/<DOC>.json files under {input_dir}")
    total = 0
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records: List[dict] = data if isinstance(data, list) else data.get("occurrences", [])
        total += lemmatise_records(records, nlp, overwrite=args.overwrite)
        out = output_dir / path.relative_to(input_dir)
        if isinstance(data, list):
            data = {"occurrences": data}
        write_doc(out, data)
    print(f"[OK] {len(files)} file(s), {total} target lemma(s) written with {args.spacy_model} -> {output_dir}")


def _run_glossary(args: argparse.Namespace) -> None:
    nlp = load_model(args.spacy_model)
    glossary = Path(args.glossary)
    output = Path(args.output) if args.output else glossary.with_name(glossary.stem + ".fr_lemmas.json")
    prefs = set()
    with open(glossary, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            fr = (row.get("skos:prefLabel@fr") or "").strip()
            if fr:
                prefs.add(fr)
    mapping = {fr: lemmatise(nlp, fr) for fr in sorted(prefs)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"spacy_model": args.spacy_model, "lemmas": mapping}, f, ensure_ascii=False, indent=2)
    print(f"[OK] {len(mapping)} prefLabel_fr lemmatised with {args.spacy_model} -> {output}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m termjudge.lemmas",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)
    o = sub.add_parser("occurrences", help="Fill 'target_lemma' in occurrence files.")
    o.add_argument("--input-dir", required=True, help="Directory of <SYSTEM>/<DOC>.json files.")
    o.add_argument("--output-dir", default=None,
                   help="Where to write the completed files (same layout). Default: in place.")
    o.add_argument("--overwrite", action="store_true",
                   help="Recompute lemmas even when 'target_lemma' is already present.")
    o.add_argument("--spacy-model", default=DEFAULT_MODEL)
    g = sub.add_parser("glossary", help="Write the FR-lemma sidecar of a SKOS glossary.")
    g.add_argument("--glossary", required=True, help="SKOS glossary CSV (';'-separated).")
    g.add_argument("--output", default=None,
                   help="Sidecar path. Default: <glossary_stem>.fr_lemmas.json next to the glossary.")
    g.add_argument("--spacy-model", default=DEFAULT_MODEL)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.command == "occurrences":
        _run_occurrences(args)
    elif args.command == "glossary":
        _run_glossary(args)
    else:  # pragma: no cover
        print(f"Unknown command {args.command!r}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

# Input format

TermJudge evaluates the translation of term occurrences. It reads three inputs:

1. **occurrence files**, one JSON file per (system, document), listing the aligned term occurrences;
2. **document texts**, one file per document on the source side and one per (system, document) on the target side, one segment per line;
3. optionally a **glossary** in SKOS CSV format.

All paths are given on the command line; the occurrence files contain no path.

```
<input-dir>/<SYSTEM>/<DOC>.json          occurrence files (--input-dir)
<source-docs-dir>/<DOC>.txt              source documents (--source-docs-dir)
<target-docs-dir>/<SYSTEM>/<DOC>.txt     system outputs (--target-docs-dir)
<glossary>.csv                           glossary (--glossary), with an optional
<glossary>.fr_lemmas.json                lemma sidecar next to it
```

`<SYSTEM>` and `<DOC>` are free identifiers; they must match between the three inputs.

## Occurrence files

```json
{
  "doc_id": "doc12",
  "system": "sysA",
  "occurrences": [
    {
      "term_id": 658,
      "head_term": "the list experiment",
      "source_term": "the List Experiment",
      "source_canonical": "the list experiment",
      "source_lemma": "the list experiment",
      "variation_category": "no_variation",
      "source_segment_ids": [0],
      "target_segment_ids": [0],
      "target_term": "expérience de la liste",
      "target_canonical": "expérience de la liste",
      "target_lemma": "expérience de le liste",
      "source_segment": "An Application of the List Experiment to Estimate Abortion Prevalence in Karachi, Pakistan",
      "target_segment": "Application de l’expérience de la liste à l’estimation de la prévalence de l’avortement à Karachi (Pakistan)"
    }
  ]
}
```

`doc_id` and `system` are optional; when present they must match the file name and the directory name. A bare JSON list of occurrences is also accepted. Occurrences are listed in reading order (by segment, then by position in the segment): the order of the list is the order the metric uses to build each concept's history.

The smallest valid occurrence has three fields:

```json
{"source_term": "decoder", "source_segment_ids": [1], "target_term": "décodeur"}
```

### Fields

| Field | Status | Type | Meaning and fallback |
|---|---|---|---|
| `source_term` | required | string | The source form as it appears in the segment (case preserved). Shown to the judges. |
| `source_segment_ids` | required | list of int | Index of the segment(s) containing the occurrence, 0-based, into the source document (line numbers of `<DOC>.txt`). |
| `target_term` | required | string | The target span aligned with the source term, as produced by the system. An empty string means the term was not translated. Shown to the judges and used for the divergence match when no lemma is given. |
| `head_term` | optional | string | Head term (preferred label) of the concept. Groups the occurrences of a concept and tells the consistency judge whether the source form is the head term or a variant. Fallback: `source_term` itself (each distinct source form is then its own concept, and the prompt receives "(unknown)" for head term or variant). |
| `term_id` | optional | int | Identifier of the concept's glossary entry (column `tid` of the glossary). Fallback: lookup by `head_term` (or `source_term`) among the glossary's English preferred and alternative labels. |
| `source_canonical` | optional | string | Canonical (normalised) source form. Fallback: `source_term`. |
| `source_lemma` | optional | string | Lemmatised source form. Occurrences of one concept are grouped by this value, so singular/plural/case variants of a source term form one chain. Fallback: `source_canonical`, then `source_term`, normalised (lowercase, whitespace, edge punctuation). |
| `variation_category` | optional | string | How the source form relates to the head term: `no_variation`, `VG` (graphical), `VR` (reduction), `VE` (expansion), `VL` (lexical), `VMS` (morphosyntactic), `CM` (combined). See below. Fallback: unknown. |
| `target_segment_ids` | optional | list of int | Segment index on the target side. Fallback: `source_segment_ids` (source and target segmentations are aligned). |
| `target_canonical` | optional | string | Canonical target form (shown in the chains). Fallback: `target_term`. |
| `target_lemma` | optional | string | Lemma of the target span, compared with the lemma of the reference to detect divergences. Fallback: `target_canonical`, then `target_term`. Can be filled by `python -m termjudge.lemmas occurrences` (see below). |
| `source_segment` | optional | string | Text of the source segment. Fallback: the line(s) `source_segment_ids` of the source document (requires `--source-docs-dir`). |
| `target_segment` | optional | string | Text of the target segment. Fallback: the line(s) `target_segment_ids` of the system output (requires `--target-docs-dir`). |

Any other field is ignored.

### Fields that change the verdicts

Three optional fields come, in the paper, from the Concordancer preprocessing. TermJudge runs without them, but its verdicts are then not strictly those of the paper's configuration:

- **`variation_category`** decides whether the glossary translation can serve as the reference of a source form: the glossary `skos:prefLabel@fr` is the reference only for the head term (`no_variation`, or a source form equal to `skos:prefLabel@en`); variants get a reference from the document or from the LLM. The category is also shown to the LLM when it generates a reference. Without it, a source form is treated as the head term only when it equals the glossary's English preferred label, and the reference generation prompt reads "(unknown)".
- **`source_lemma` and `target_lemma`** make the grouping of occurrences and the divergence match insensitive to inflection. Without them, raw forms are compared after a light normalisation (case, whitespace, edge punctuation, leading French articles, and for the "graphic" level hyphens and diacritics), so "réseau de neurones" and "réseaux de neurones" count as a divergence that the judges then have to settle. The `[lemmas]` extra fills `target_lemma` and the glossary sidecar with spaCy, with the model used in the paper.
- **`term_id` and `head_term`** link an occurrence to its glossary entry and to the other occurrences of the same concept. Without `term_id`, the entry is looked up by name; without `head_term`, each source form is a concept of its own, and the consistency judge no longer sees the renderings of the concept's other forms.

## Document texts

One file per document, UTF-8, one segment per line: line `i` is segment `i` of `source_segment_ids` / `target_segment_ids`. The system output of each document must follow the segmentation of the source. These files feed the context window of the judges (`--context-window N` injects the `N` preceding segment pairs into the prompts), head the chain views, and fill missing `source_segment` / `target_segment` values.

## Glossary

A CSV file with `;` as separator and `##` between several labels in one cell:

```
tid;skos:prefLabel@en;skos:prefLabel@fr;skos:altLabel@en;skos:altLabel@fr
0;machine translation;traduction automatique;MT;TA
1;neural network;réseau de neurones;neural net;réseau neuronal
2;attention mechanism;mécanisme d'attention;attention;
```

- `tid` is optional; without it, the row index is the entry's identifier.
- `skos:prefLabel@en` is the English head term; `skos:altLabel@en` lists its English variants. Both are used to find an entry by name when a record has no `term_id`.
- `skos:prefLabel@fr` is the French head term: the reference against which the occurrences of the head term are matched, and the established term shown to the error judge.
- `skos:altLabel@fr` lists admitted French synonyms: the consistency judge is told whether a divergent rendering is one of them.
- Further columns (definitions, examples) are ignored.

The glossary is optional. Without one, `--reference-mode frequency` or `first` takes the reference from the document itself (most frequent, or first, rendering of the source form by the system), and `--reference-mode generate` or `validated` lets the LLM produce it from the source-side contexts.

### Lemma sidecar

The French labels of a glossary are display forms, not lemmas. To match them against lemmatised target spans, TermJudge reads a sidecar file next to the glossary, `<glossary_stem>.fr_lemmas.json`, mapping each `skos:prefLabel@fr` to its lemma:

```json
{"spacy_model": "fr_core_news_lg", "lemmas": {"Facteurs de risque": "facteur de risque"}}
```

The sidecar must come from the same lemmatiser as the `target_lemma` values. Without it, a warning is printed and the raw French label is used.

## Lemmatisation with spaCy

```
pip install "termjudge[lemmas]"
python -m spacy download fr_core_news_lg
python -m termjudge.lemmas occurrences --input-dir alignments/ --output-dir alignments_lemmatised/
python -m termjudge.lemmas glossary --glossary glossary.csv
```

The first command fills `target_lemma` in every occurrence file (in place when `--output-dir` is omitted); the second writes the glossary sidecar. Both use `fr_core_news_lg`, the model of the paper.

## Converting Concordancer output

Users of the Concordancer pipeline can convert its alignment files with:

```
python scripts/from_concordancer.py \
    --input-dir <step4 output>/free_llm_output \
    --output-dir data/mycorpus/alignments \
    --targets-dir data/mycorpus/targets
```

The script keeps the fields listed above under their public names (`src_term` becomes `source_term`, `lemmas_list` becomes `source_lemma`, `tgt_phrase_final_form_spacy_lemma` becomes `target_lemma`, and so on) and copies the system outputs referenced by the records. The source documents are the Concordancer's `source_docs/<DOC>.txt` files, to copy as such.

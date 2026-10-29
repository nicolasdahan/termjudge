# BioMQM-Terms: datasheet

BioMQM-Terms is a terminology evaluation resource for English-to-French machine translation in the biomedical domain. It adds, to the English-French portion of the bio-MQM corpus, a glossary of the biomedical concepts of its 50 documents and, for each of its ten translations, the aligned occurrences of these concepts, in the input format of TermJudge. It was built for the paper *TermJudge: A Document-Level Metric Judging, Not Counting, Terminology in Machine Translation Evaluation* (WMT 2026).

## Contents

```
data/biomqm/
  glossary/biomqm_glossary.csv             SKOS glossary, 710 concepts (EN-FR)
  glossary/biomqm_glossary.fr_lemmas.json  lemmas of the French head terms (spaCy fr_core_news_lg)
  source/<DOC>.txt                         50 source documents, one segment per line
  targets/<SYSTEM>/<DOC>.txt               outputs of the 10 systems, same segmentation
  alignments/<SYSTEM>/<DOC>.json           aligned term occurrences, 1,320 per system
  run_termjudge.sh                         the command of the paper
  LICENSE-bio-mqm, NOTICE-bio-mqm          license and notice of the bio-MQM corpus
```

Systems: `Huawei_AGI_run1`, `Huawei_AGI_run2`, `Huawei_AGI_run3`, `LISN_run1`, `LISN_run2`, `LISN_run3`, `TMT_run1`, `mtcarousel` (the eight MT submissions to the WMT21 biomedical task kept in bio-MQM for this language pair) and the two human reference translations `reference` and `reference_r1`.

## Statistics

| | |
|---|---|
| Language pair, domain | English to French, biomedical (abstracts of scientific articles) |
| Documents | 50 |
| Source segments | 384 (7,536 words); 373 contain at least one term occurrence |
| Systems | 10 (8 MT systems, 2 human references) |
| Glossary entries | 710 concepts, 189 with at least one English variant (256 variants in all) |
| Concepts with occurrences | 659 |
| Term occurrences per system | 1,320 (13,200 term-translation pairs in all) |
| Distinct (document, source form) chains | 978, of which 782 are hapax (a single occurrence) |
| Source variation categories | no variation 979, graphical 98, reduction 92, expansion 76, lexical 44, combined 17, morphosyntactic 14 |
| Empty alignments (term not found in the output) | 67 of 13,200 |

The occurrences are identical for the ten systems: term detection operates on the source text, and each system's file differs only by the aligned target spans and the target segments.

## Provenance

**Texts and translations.** The documents, the MT outputs and the two reference translations are those of the bio-MQM corpus (Zouhar, Ding, Currey, Badeka, Wang and Thompson, *Fine-Tuned Machine Translation Metrics Struggle in Unseen Domains*, 2024; https://github.com/amazon-science/bio-mqm-dataset), English-French portion, copied as such and segmented as in the corpus. bio-MQM is distributed under the Apache License 2.0; its license and notice are reproduced in this directory. The MQM error annotations of bio-MQM are not included: they are the human ground truth of the paper's meta-evaluation, which this repository does not cover.

**Glossary.** The English terms were extracted manually from the 50 source documents by exhaustive reading, independently of their involvement in any translation error, then revised and deduplicated; variants (acronyms, synonyms, alternative forms) are listed as alternative labels. The French labels come from the bilingual MeSH thesaurus of the U.S. National Library of Medicine (https://www.nlm.nih.gov/mesh/) wherever a concept could be matched to a MeSH descriptor, by its preferred label, one of its alternative labels, a singular/plural variant or a hyphenation variant. The remaining concepts, too specific or too compositional for MeSH, were translated manually with the terminology banks TERMIUM Plus (https://www.btb.termiumplus.gc.ca) and IATE (https://iate.europa.eu) and the French edition of Wikipedia (https://fr.wikipedia.org). For each entry, `skos:prefLabel@en` and `skos:prefLabel@fr` are the head terms and `skos:altLabel@en` lists the English variants; the column `tid` is the identifier that the occurrence files use (`term_id`).

**Term occurrences.** Occurrences of the glossary concepts were detected in the source documents with the Concordancer (https://gitlab.inria.fr/almanach/concordancer), which recognises, beyond the glossary labels and their lemmas, variants derived from the entries or detected in context. Each source form was labelled with its variation category relative to the head term (no variation, graphical, morphosyntactic, reduction, expansion, lexical, combined) with `gpt-4.1` and the prompt reproduced in `docs/concordancer_prompts/label_variation.txt`. Each occurrence was then aligned with its translation in the target segment by `gpt-4.1-mini` (temperature 0, at most 20 output tokens), with the six-example prompt reproduced in `docs/concordancer_prompts/align_term.txt`, instructed to copy the translation of the term verbatim from the target segment; an empty span means the term was not found in the output. Aligned spans were normalised for case, punctuation and whitespace and lemmatised with spaCy (`fr_core_news_lg`), which gives the `target_canonical` and `target_lemma` fields; `source_lemma` comes from the Concordancer's per-token lemmas. The paper reports a manual validation of the same preprocessing on another corpus (term detection precision 91.5% and recall 95.5%, alignment correct for 78 of 80 sampled occurrences).

## Format

The occurrence files follow the TermJudge input format documented in `docs/input_format.md`: for each occurrence, the glossary identifier and head term (`term_id`, `head_term`), the source form, its canonical form, its lemma and its variation category, the segment indices, the aligned target span with its canonical form and lemma, and the source and target segments. They were produced from the Concordancer output by `scripts/from_concordancer.py`.

## Intended use

Evaluating the terminology of English-to-French MT systems on these documents with TermJudge (`run_termjudge.sh` runs the configuration of the paper), comparing other terminology metrics on the same occurrences, or studying terminological variation in biomedical translation. The human MQM annotations needed to reproduce the paper's meta-evaluation are available from the bio-MQM repository, and the occurrences carry the segment indices that link them to the bio-MQM segments.

## Known limitations

- The alignments and the variation categories were produced by LLMs and were not exhaustively checked; the validation figures above were measured on a sample of another corpus.
- The glossary has English variants for 189 concepts but French synonyms (`skos:altLabel@fr`) for only two: the consistency judge almost never sees admitted synonyms on this corpus.
- The resource covers one language pair, one domain and ten systems, and the glossary was built from these 50 documents only.

## License

The texts and translations derived from bio-MQM remain under the Apache License 2.0 of that corpus (`LICENSE-bio-mqm`, `NOTICE-bio-mqm`). The MeSH-derived French labels of the glossary are subject to the terms of use of the U.S. National Library of Medicine (https://www.nlm.nih.gov/databases/download/terms_and_conditions.html), which require that MeSH be credited as their source. The glossary structure, the manually produced labels, the variation categories and the alignments are released under the MIT license of this repository.

If you use BioMQM-Terms, please cite the TermJudge paper and the bio-MQM paper:

```
@inproceedings{zouhar-etal-2024-fine,
  title     = {Fine-Tuned Machine Translation Metrics Struggle in Unseen Domains},
  author    = {Zouhar, Vil{\'e}m and Ding, Shuoyang and Currey, Anna and Badeka, Tatyana and Wang, Jenyuan and Thompson, Brian},
  editor    = {Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  month     = aug,
  year      = {2024},
  address   = {Bangkok, Thailand},
  publisher = {Association for Computational Linguistics},
  pages     = {488--500},
  doi       = {10.18653/v1/2024.acl-short.45},
  url       = {https://aclanthology.org/2024.acl-short.45/}
}
```

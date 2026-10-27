import csv

from termjudge.metric.annotation import OccurrenceAnnotation
from termjudge.metric.match import any_equivalence, match_level, normalise, normalise_graphic
from termjudge.metric.scoring import (
    DEFAULT_PENALTIES, load_penalties, score_document, write_aggregated_csv, write_per_document_csv,
)


def test_normalise_strips_case_articles_and_edges():
    assert normalise("  L'avortement, ") == "avortement"
    assert normalise("Les Facteurs de risque") == "facteurs de risque"
    assert normalise("") == ""
    assert normalise_graphic("néphro-lithotomie") == "nephro lithotomie"


def test_match_levels():
    assert match_level("prévalence de l'avortement", "la prévalence de l'avortement") == "exact"
    assert match_level("nephrolithotomie percutanee", "néphrolithotomie percutanée") == "graphic"
    assert match_level("avortement", "IVG") == "none"
    assert match_level("", "avortement") == "none"
    assert any_equivalence("Réseau neuronal", ["réseau de neurones", "reseau neuronal"])
    assert not any_equivalence("réseau", ["réseau de neurones"])


def _ann(verdict, is_hapax=False):
    return OccurrenceAnnotation(
        system="s", doc_id="d", term_id=0, concept_key="c", source_term="x",
        target_canonical="y", reference="y", reference_source="glossary",
        reference_confidence="high", match_level="none", is_first_mention=False,
        variation_category="", verdict=verdict, is_hapax=is_hapax,
    )


def test_default_penalties_follow_the_paper():
    assert DEFAULT_PENALTIES["CONFORME"] == 0 and DEFAULT_PENALTIES["VAR_J"] == 0
    assert DEFAULT_PENALTIES["VAR_NJ"] == 0.5
    assert DEFAULT_PENALTIES["A1"] == 1 and DEFAULT_PENALTIES["A3"] == 3 and DEFAULT_PENALTIES["A4"] == 4
    assert DEFAULT_PENALTIES["C"] == 4 and DEFAULT_PENALTIES["D"] == 3
    assert DEFAULT_PENALTIES["E"] == 1 and DEFAULT_PENALTIES["F"] == 1
    assert DEFAULT_PENALTIES["G"] == 1 and DEFAULT_PENALTIES["H"] == 1
    assert DEFAULT_PENALTIES["SKIPPED_HAPAX"] == 0


def test_score_document_is_the_mean_penalty():
    anns = [_ann("CONFORME"), _ann("A4"), _ann("VAR_NJ"), _ann("VAR_J")]
    s = score_document(anns, DEFAULT_PENALTIES)
    assert s["n_occurrences"] == 4
    assert s["total_penalty"] == 4.5
    assert s["score"] == 1.125
    assert s["distribution"] == {"CONFORME": 1, "A4": 1, "VAR_NJ": 1, "VAR_J": 1}
    assert score_document([], DEFAULT_PENALTIES)["score"] == 0.0


def test_penalty_override_and_csv(tmp_path):
    pen = load_penalties('{"VAR_NJ": 2, "A4": 1}')
    assert pen["VAR_NJ"] == 2 and pen["A4"] == 1 and pen["A1"] == 1
    rows = [{"doc_id": "d1", **score_document([_ann("A4"), _ann("CONFORME")], pen)},
            {"doc_id": "d2", **score_document([_ann("VAR_NJ")], pen)}]
    write_per_document_csv(tmp_path / "per_doc.csv", rows)
    write_aggregated_csv(tmp_path / "agg.csv", rows, "sysA")
    per_doc = list(csv.DictReader(open(tmp_path / "per_doc.csv", encoding="utf-8")))
    assert [r["doc_id"] for r in per_doc] == ["d1", "d2"]
    assert per_doc[0]["score"] == "0.5" and per_doc[0]["n_A4"] == "1"
    agg = list(csv.DictReader(open(tmp_path / "agg.csv", encoding="utf-8")))[0]
    assert agg["system"] == "sysA" and agg["n_documents"] == "2"
    assert float(agg["mean_score"]) == 1.25

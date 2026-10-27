import json

import pytest

from termjudge.io import records as io_records
from termjudge.io.records import RecordError


def _write(tmp_path, system, doc_id, payload):
    d = tmp_path / system
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{doc_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return d


FULL = {
    "term_id": 1, "head_term": "neural network", "source_term": "neural networks",
    "source_canonical": "neural networks", "source_lemma": "neural network",
    "variation_category": "no_variation", "source_segment_ids": [7], "target_segment_ids": [7],
    "target_term": "réseaux de neurones", "target_canonical": "réseaux de neurones",
    "target_lemma": "réseau de neurone",
    "source_segment": "We evaluate neural networks.", "target_segment": "Nous évaluons des réseaux de neurones.",
}

MINIMAL = {"source_term": "decoder", "source_segment_ids": [1], "target_term": "décodeur"}


def test_nominal_record(tmp_path):
    d = _write(tmp_path, "sysA", "doc1", {"doc_id": "doc1", "system": "sysA", "occurrences": [FULL]})
    recs = io_records.load_doc(d, "doc1")
    assert len(recs) == 1
    r = recs[0]
    assert r["doc_id"] == "doc1" and r["system"] == "sysA"
    assert io_records.head_term(r) == "neural network"
    assert io_records.source_lemma(r) == "neural network"
    assert io_records.canonical_source(r) == "neural networks"
    assert io_records.raw_target(r) == "réseaux de neurones"
    assert io_records.canonical_target(r) == "réseaux de neurones"
    assert io_records.target_lemma(r) == "réseau de neurone"
    assert io_records.target_segment_ids(r) == [7]
    assert io_records.list_systems(tmp_path) == ["sysA"]
    assert io_records.list_docs(d) == ["doc1"]


def test_optional_fields_fall_back(tmp_path):
    d = _write(tmp_path, "sysA", "doc1", [MINIMAL])   # bare list accepted
    r = io_records.load_doc(d, "doc1")[0]
    assert io_records.head_term(r) == ""
    assert io_records.source_lemma(r) == "decoder"
    assert io_records.canonical_source(r) == "decoder"
    assert io_records.canonical_target(r) == "décodeur"
    assert io_records.target_lemma(r) == "décodeur"
    assert io_records.target_segment_ids(r) == [1]
    # segments recovered from the document texts
    io_records.fill_segments([r], ["s0", "The decoder works."], ["c0", "Le décodeur fonctionne."])
    assert r["source_segment"] == "The decoder works."
    assert r["target_segment"] == "Le décodeur fonctionne."


def test_missing_segment_without_text_is_explicit(tmp_path):
    d = _write(tmp_path, "sysA", "doc1", [MINIMAL])
    recs = io_records.load_doc(d, "doc1")
    with pytest.raises(RecordError, match="source_segment"):
        io_records.fill_segments(recs, [], [], where="doc1.json")


def test_untranslated_target_is_allowed(tmp_path):
    d = _write(tmp_path, "sysA", "doc1", [dict(MINIMAL, target_term="")])
    r = io_records.load_doc(d, "doc1")[0]
    assert io_records.raw_target(r) == ""
    assert io_records.target_lemma(r) == ""


@pytest.mark.parametrize("field", ["source_term", "source_segment_ids", "target_term"])
def test_required_field_missing(tmp_path, field):
    rec = {k: v for k, v in MINIMAL.items() if k != field}
    d = _write(tmp_path, "sysA", "doc1", [rec])
    with pytest.raises(RecordError) as e:
        io_records.load_doc(d, "doc1")
    assert field in str(e.value)
    assert "occurrence #0" in str(e.value)


def test_malformed_values(tmp_path):
    d = _write(tmp_path, "sysA", "doc1", [dict(MINIMAL, source_segment_ids=["1"])])
    with pytest.raises(RecordError, match="source_segment_ids"):
        io_records.load_doc(d, "doc1")
    d = _write(tmp_path, "sysB", "doc1", [dict(MINIMAL, term_id="12")])
    with pytest.raises(RecordError, match="term_id"):
        io_records.load_doc(d, "doc1")


def test_doc_id_and_system_must_match_the_path(tmp_path):
    d = _write(tmp_path, "sysA", "doc1", {"doc_id": "doc2", "occurrences": [MINIMAL]})
    with pytest.raises(RecordError, match="doc_id"):
        io_records.load_doc(d, "doc1")
    d = _write(tmp_path, "sysA", "doc3", {"system": "sysZ", "occurrences": [MINIMAL]})
    with pytest.raises(RecordError, match="system"):
        io_records.load_doc(d, "doc3")


def test_write_doc_round_trip(tmp_path):
    doc = {"doc_id": "doc1", "system": "sysA", "occurrences": [FULL, MINIMAL]}
    path = tmp_path / "sysA" / "doc1.json"
    io_records.write_doc(path, doc)
    assert json.loads(path.read_text(encoding="utf-8")) == doc
    assert len(io_records.load_doc(path.parent, "doc1")) == 2

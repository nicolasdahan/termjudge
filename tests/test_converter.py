import importlib.util
import math
from pathlib import Path

from termjudge.io import records as io_records

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "from_concordancer.py"
spec = importlib.util.spec_from_file_location("from_concordancer", SCRIPT)
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)


def _rec(**kw):
    base = {
        "term_id": 3, "term": "abortion prevalence", "src_term": "Abortion Prevalence",
        "form": "abortion prevalence", "variation_words": "abortion prevalence",
        "lemmas_list": ["abortion", "prevalence"], "variation_category": "no_variation",
        "src_sent_ids": [0], "tgt_sent_ids": [0], "span_index": 1, "doc_id": "doc12",
        "src_segment": "An Application of the List Experiment to Estimate Abortion Prevalence",
        "tgt_segment": "Application de l'expérience de la liste à l'estimation de la prévalence de l'avortement",
        "tgt_phrase": "prévalence de l'avortement", "tgt_phrase_cleaned": "prévalence de l'avortement",
        "llm_free_alignment": "prévalence de l'avortement", "llm_free_alignment_raw": "prévalence de l'avortement",
        "tgt_phrase_final_form_canonical": "prévalence de l'avortement",
        "tgt_phrase_final_form_spacy_lemma": "prévalence de le avortement",
        "kind": math.nan, "has_error": False, "errors": [],
        "tgt_path": "/somewhere/doc12.txt",
    }
    base.update(kw)
    return base


def test_convert_record_keeps_the_public_fields_only():
    out = conv.convert_record(_rec())
    assert set(out) == {
        "term_id", "head_term", "source_term", "source_canonical", "source_lemma",
        "variation_category", "source_segment_ids", "target_segment_ids", "target_term",
        "target_canonical", "target_lemma", "source_segment", "target_segment",
    }
    assert out["source_term"] == "Abortion Prevalence"
    assert out["source_lemma"] == "abortion prevalence"
    assert out["target_lemma"] == "prévalence de le avortement"
    assert "tgt_path" not in out and "errors" not in out


def test_convert_record_fallbacks():
    # no lemmas, no canonical form, untranslated term
    r = _rec(lemmas_list=None, tgt_phrase_final_form_spacy_lemma=None,
             tgt_phrase_final_form_canonical=None, tgt_phrase_cleaned="", tgt_phrase="",
             llm_free_alignment="", llm_free_alignment_raw="")
    out = conv.convert_record(r)
    assert "source_lemma" not in out and "target_lemma" not in out and "target_canonical" not in out
    assert out["target_term"] == ""
    assert io_records.source_lemma(out) == "abortion prevalence"
    assert io_records.canonical_target(out) == ""


def test_convert_doc_orders_by_segment_then_span():
    recs = [_rec(src_sent_ids=[2], span_index=0, src_term="B"),
            _rec(src_sent_ids=[0], span_index=5, src_term="C"),
            _rec(src_sent_ids=[0], span_index=1, src_term="A")]
    doc = conv.convert_doc(recs, "doc12", "sysA")
    assert doc["doc_id"] == "doc12" and doc["system"] == "sysA"
    assert [o["source_term"] for o in doc["occurrences"]] == ["A", "C", "B"]


def test_convert_and_load(tmp_path):
    doc = conv.convert_doc([_rec()], "doc12", "sysA")
    io_records.write_doc(tmp_path / "sysA" / "doc12.json", doc)
    recs = io_records.load_doc(tmp_path / "sysA", "doc12")
    assert len(recs) == 1 and recs[0]["head_term"] == "abortion prevalence"

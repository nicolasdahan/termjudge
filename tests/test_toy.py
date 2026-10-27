import csv
import json
from pathlib import Path

from termjudge.metric import annotate, score
from termjudge.metric.annotate import PROMPTS_DIR, inject_domain, inject_examples, load_prompts

TOY = Path(__file__).resolve().parents[1] / "examples" / "toy"


def test_prompts_ship_with_the_package():
    prompts = load_prompts()
    assert set(prompts) == {"select_reference", "generate_reference", "q1_classify", "q2_vartj", "hapax_judge"}
    assert prompts["hapax_judge"] == prompts["q1_classify"]
    assert (PROMPTS_DIR / "select_reference.examples.txt").is_file()
    resolved = inject_domain(inject_examples(prompts), "biomedical")
    for name, text in resolved.items():
        assert "{examples_block}" not in text and "{domain}" not in text, name
        assert "biomedical domain" in text
    assert "<examples>" in resolved["select_reference"]
    assert "<examples>" not in resolved["q1_classify"]


def _run_dry(tmp_path, extra):
    annotate.main([
        "--input-dir", str(TOY / "alignments"),
        "--source-docs-dir", str(TOY / "source"),
        "--target-docs-dir", str(TOY / "targets"),
        "--context-window", "5", "--domain", "NLP",
        "--output-dir", str(tmp_path), "--run-tag", "t", "--dry-run", "--no-progress",
        *extra,
    ])


def test_toy_dry_run_with_glossary(tmp_path):
    _run_dry(tmp_path, ["--glossary", str(TOY / "glossary.csv"), "--reference-mode", "validated"])
    for system in ("sysA", "sysB"):
        anns = json.load(open(tmp_path / system / "runs" / "t" / "annotations" / "doc1.json", encoding="utf-8"))
        assert len(anns) == 10
        assert {a["verdict"] for a in anns} <= {"CONFORME", "VAR_NJ", "SKIPPED_HAPAX"}
        assert (tmp_path / system / "runs" / "t" / "chains" / "doc1.md").is_file()
        assert (tmp_path / system / "runs" / "t" / "prompts_used" / "q1_classify.txt").is_file()
    by_src = {(a["source_term"], a["source_segment_ids"][0]): a
              for a in json.load(open(tmp_path / "sysB" / "runs" / "t" / "annotations" / "doc1.json", encoding="utf-8"))}
    # glossary head term reference, exact match on lemmas (plural)
    a = by_src[("neural networks", 7)]
    assert a["reference"] == "réseau de neurones" and a["reference_source"] == "glossary"
    assert a["match_level"] == "exact" and a["verdict"] == "CONFORME"
    # divergence without an LLM -> VAR_NJ by default
    a = by_src[("neural network", 0)]
    assert a["target_term"] == "réseau neuronal" and a["verdict"] == "VAR_NJ"
    # hapax without an LLM -> skipped
    a = by_src[("Beam search", 4)]
    assert a["is_hapax"] and a["verdict"] == "SKIPPED_HAPAX"
    # scoring
    score.main(["--annotations-dir", str(tmp_path), "--run-tag", "t"])
    rows = list(csv.DictReader(open(tmp_path / "sysB" / "runs" / "t" / "scores_per_document.csv", encoding="utf-8")))
    assert rows[0]["doc_id"] == "doc1" and rows[0]["n_occurrences"] == "10"
    agg = list(csv.DictReader(open(tmp_path / "sysA" / "runs" / "t" / "scores_aggregated.csv", encoding="utf-8")))[0]
    assert agg["system"] == "sysA" and agg["n_documents"] == "1"


def test_toy_dry_run_without_glossary(tmp_path):
    _run_dry(tmp_path, ["--reference-mode", "frequency"])
    anns = json.load(open(tmp_path / "sysA" / "runs" / "t" / "annotations" / "doc1.json", encoding="utf-8"))
    assert len(anns) == 10
    assert all(a["concept_preflabel_fr"] == "" for a in anns)
    assert {a["reference_source"] for a in anns} <= {"frequency", "first"}
    by_src = {(a["source_term"], a["source_segment_ids"][0]): a for a in anns}
    assert by_src[("decoder", 2)]["reference"] == "décodeur"
    assert by_src[("decoder", 2)]["verdict"] == "CONFORME"

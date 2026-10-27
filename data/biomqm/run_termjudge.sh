#!/usr/bin/env bash
# TermJudge on BioMQM-Terms, in the configuration of the paper: ten systems,
# fifty documents, gpt-4.1-mini as the judge. Runs from any directory; the
# outputs go to <repository>/output/biomqm/<SYSTEM>/runs/paper/.
#
# Requires OPENAI_API_KEY, or BASE_URL and MODEL to use another judge through
# an OpenAI-compatible endpoint (the verdicts then differ from the paper's).
# About 12,000 judge calls on the first run; later runs are served by the cache.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${OUTPUT_DIR:-$ROOT/output/biomqm}"

termjudge-annotate \
  --input-dir "$HERE/alignments" \
  --glossary "$HERE/glossary/biomqm_glossary.csv" \
  --source-docs-dir "$HERE/source" \
  --target-docs-dir "$HERE/targets" \
  --reference-mode validated \
  --q1-mode llm --q2-mode llm --hapax-mode evaluate \
  --judge-path auto --llm-shortcuts non_hapax \
  --propagate-q1 on --propagate-pick last \
  --context-window 5 --domain biomedical \
  --model "${MODEL:-gpt-4.1-mini}" ${BASE_URL:+--base-url "$BASE_URL"} \
  --max-tokens 256 --workers "${WORKERS:-8}" \
  --output-dir "$OUT" --run-tag paper "$@"

termjudge-score --annotations-dir "$OUT" --run-tag paper
termjudge-score --annotations-dir "$OUT" --run-tag paper --exclude-hapax
echo "Results under $OUT/<SYSTEM>/runs/paper/"

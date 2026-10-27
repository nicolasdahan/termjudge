#!/usr/bin/env bash
# Smoke test on the toy example. Runs from any directory.
#
#   examples/toy/run_toy.sh [dry|full] [OUTPUT_DIR]
#
#   dry  (default): no LLM call, no API key needed
#   full          : real judge calls, on the OpenAI API (OPENAI_API_KEY) or on
#                   any OpenAI-compatible endpoint given by BASE_URL
#
# Environment for the full mode: MODEL (default gpt-4.1-mini) and BASE_URL
# (unset = OpenAI API). OUTPUT_DIR defaults to examples/toy/output.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-dry}"
OUT="${2:-$HERE/output}"

COMMON=(
  --input-dir "$HERE/alignments"
  --glossary "$HERE/glossary.csv"
  --source-docs-dir "$HERE/source"
  --target-docs-dir "$HERE/targets"
  --context-window 5
  --domain NLP
  --reference-mode validated
  --output-dir "$OUT"
  --no-progress
)

case "$MODE" in
  dry)
    termjudge-annotate "${COMMON[@]}" --run-tag dry-run --dry-run
    termjudge-score --annotations-dir "$OUT" --run-tag dry-run
    ;;
  full)
    termjudge-annotate "${COMMON[@]}" --run-tag full --save-prompts auto \
      --model "${MODEL:-gpt-4.1-mini}" ${BASE_URL:+--base-url "$BASE_URL"}
    termjudge-score --annotations-dir "$OUT" --run-tag full
    ;;
  *)
    echo "usage: $0 [dry|full] [OUTPUT_DIR]" >&2; exit 2 ;;
esac
echo "Results under $OUT/<SYSTEM>/runs/"

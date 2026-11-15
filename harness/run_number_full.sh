#!/usr/bin/env bash
# Full 360-number-question run on a given pool with the adopted number policy (10-item map batches, no rank prior, gpt-5.4-mini reduce,
# deterministic number formatter in the pipeline). MAP_MODEL / REDUCE_REASONING pass through from the environment.
# The judge step only runs the deterministic em metric here (no open_end questions), so it is free. usage: run_number_full.sh NAME POOL
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
PY="$HOME/atmbench/.venv/bin/python"; QA="$SP/atm/atm-bench-number.json"
export ITEMS_FILE="$SP/atm/items-san.json" BEST_GUESS=1 MAP_MODEL="${MAP_MODEL:-deepseek/deepseek-v4.1-flash}" FALLBACK_MODEL=anthropic/claude-sonnet-5.5 ANSWER_MODEL=openai/gpt-5.4-mini
NAME="$1"; POOL="$2"
echo "=== $NAME pool=$POOL map=$MAP_MODEL reduce_reasoning=${REDUCE_REASONING:-minimal} ($(date -u +%H:%M) UTC)"
RANK_PRIOR_K=0 RANK_PRIOR_K_LIST=10 "$PY" -u mf_answer_r2.py --qa "$QA" --pool "$POOL" --topk 25 --batch 10 --mode mapreduce --workers 6 --out "ans/$NAME.jsonl" 2> "runs/$NAME.err" | tail -1
echo "answers $(wc -l < "ans/$NAME.jsonl"); FAILED $(grep -c '^FAILED' "runs/$NAME.err"); $(tail -1 "runs/$NAME.err")"
WORKERS=6 ./judge.sh "$NAME" "$QA" | tee -a runs/number_full_result.txt
echo "=== NUMBER FULL DONE ($(date -u +%H:%M) UTC)"

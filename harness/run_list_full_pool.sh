#!/usr/bin/env bash
# Full 139-list-question run on a given pool with the adopted list policy (one 25-item map call, K_LIST default 10).
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
PY="${ATM_PY:-$HOME/eval-archive/atm-run/.venv/bin/python}"; QA="$SP/atm/atm-bench-list-only.json"
export ITEMS_FILE="$SP/atm/items-san.json" BEST_GUESS=1 MAP_MODEL="${MAP_MODEL:-deepseek/deepseek-v4.1-flash}" FALLBACK_MODEL=anthropic/claude-sonnet-5.5 ANSWER_MODEL=openai/gpt-5.4-mini
NAME="$1"; POOL="$2"; KL="${3:-10}"
echo "=== $NAME pool=$POOL K_LIST=$KL ($(date -u +%H:%M) UTC)"
RANK_PRIOR_K=0 RANK_PRIOR_K_LIST=$KL "$PY" -u mf_answer_r2.py --qa "$QA" --pool "$POOL" --topk 25 --batch 25 --mode mapreduce --workers 6 --out "ans/$NAME.jsonl" 2> "runs/$NAME.err" | tail -1
echo "answers $(wc -l < "ans/$NAME.jsonl"); FAILED $(grep -c '^FAILED' "runs/$NAME.err"); $(tail -1 "runs/$NAME.err")"
WORKERS=6 ./judge.sh "$NAME" "$QA" | tee -a runs/list_full_qr.txt
echo "=== LIST FULL DONE ($(date -u +%H:%M) UTC)"

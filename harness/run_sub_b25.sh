#!/usr/bin/env bash
# usage: run_sub_b25.sh NAME RANK [BATCH] — sha1-subset answer pass with the adopted type policy and a map batch size, then judge
set -uo pipefail; SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
NAME="$1"; RANK="$2"; B="${3:-25}"; QA="$SP/atm/atm-subset-sha1.json"; PY="${ATM_PY:-$HOME/eval-archive/atm-run/.venv/bin/python}"
export ITEMS_FILE="$SP/atm/items-san.json" BEST_GUESS=1 MAP_MODEL=deepseek/deepseek-v4.1-flash FALLBACK_MODEL=anthropic/claude-sonnet-5.5 ANSWER_MODEL=openai/gpt-5.4-mini
echo "=== SUB b$B $NAME rank=$RANK ($(date -u +%H:%M) UTC)"
RANK_PRIOR_K=0 RANK_PRIOR_K_LIST=10 "$PY" -u mf_answer_r2.py --qa "$QA" --pool "$RANK" --topk 25 --batch "$B" --mode mapreduce --workers 6 --out "ans/$NAME.jsonl" 2> "runs/$NAME.err" | tail -1
echo "answers $(wc -l < "ans/$NAME.jsonl"); FAILED $(grep -c '^FAILED' "runs/$NAME.err"); $(tail -1 "runs/$NAME.err")"
WORKERS=6 ./judge.sh "$NAME" "$QA" "" | tee -a runs/sub_b25_result.txt

#!/usr/bin/env bash
# Number-only reduce variants on the hash subset's 70 number questions (deterministic EM, no judge calls).
# usage: run_num_sub.sh NAME 'ENV=VAL ...'   e.g. run_num_sub.sh nsub_rm 'REDUCE_REASONING={"effort":"medium"}'
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
PY="${ATM_PY:-$HOME/eval-archive/atm-run/.venv/bin/python}"; QA="${QA:-$SP/atm/atm-subset-sha1-number.json}"
export ITEMS_FILE="$SP/atm/items-san.json" BEST_GUESS=1 MAP_MODEL=deepseek/deepseek-v4.1-flash FALLBACK_MODEL=anthropic/claude-sonnet-5.5 ANSWER_MODEL=openai/gpt-5.4-mini
NAME="$1"; shift
echo "=== $NAME $* ($(date +%H:%M))"
env "$@" "$PY" -u mf_answer_v2.py --qa "$QA" --pool runs/san_fused.rank.json --topk 25 --mode mapreduce --workers 6 --out "ans/$NAME.jsonl" 2> "runs/$NAME.err" | tail -1
echo "answers $(wc -l < "ans/$NAME.jsonl"); FAILED $(grep -c '^FAILED' "runs/$NAME.err"); fallbacks $(grep -c 'reduce fallback' "runs/$NAME.err")"
WORKERS=6 ./judge.sh "$NAME" "$QA" | tee -a runs/num_sub.txt

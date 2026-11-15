#!/usr/bin/env bash
# R2 subset stage: rank-prior variants V2 (top-5) and V3 (top-5, list top-10) on the 196-q sha1 subset, one judge pass each.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
PY="$HOME/atmbench/.venv/bin/python"; QA="$SP/atm/atm-subset-sha1.json"
export ITEMS_FILE="$SP/atm/items-san.json" BEST_GUESS=1 MAP_MODEL=deepseek/deepseek-v4.1-flash FALLBACK_MODEL=anthropic/claude-sonnet-5.5 ANSWER_MODEL=openai/gpt-5.4-mini
for V in "V2 5 5" "V3 5 10"; do
  set -- $V; NAME="r2_sub_$1"
  echo "=== $NAME K=$2 K_LIST=$3 ($(date +%H:%M))"
  RANK_PRIOR_K=$2 RANK_PRIOR_K_LIST=$3 "$PY" -u mf_answer_r2.py --qa "$QA" --pool runs/san_fused.rank.json --topk 25 --mode mapreduce --workers 6 --out "ans/$NAME.jsonl" 2> "runs/$NAME.err" | tail -1
  echo "answers $(wc -l < "ans/$NAME.jsonl"); reduce fallbacks $(grep -c 'reduce fallback' "runs/$NAME.err"); FAILED $(grep -c '^FAILED' "runs/$NAME.err"); $(tail -1 "runs/$NAME.err")"
  WORKERS=6 ./judge.sh "$NAME" "$QA" | tee -a runs/r2_subset.txt
done
echo "=== R2 SUBSET DONE ($(date +%H:%M))"

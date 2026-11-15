#!/usr/bin/env bash
# usage: run_list_sub_reducer.sh MODEL... — reduce-model probe on the adopted list configuration (4B-fused pool, one 25-item
# map call, K_LIST 10), hash subset 30 list q (deterministic Jaccard). The map is cached; only the reduce calls are new.
set -uo pipefail; SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
PY="$HOME/atmbench/.venv/bin/python"; QA="$SP/atm/atm-subset-sha1-list.json"
export ITEMS_FILE="$SP/atm/items-san.json" BEST_GUESS=1 MAP_MODEL=deepseek/deepseek-v4.1-flash FALLBACK_MODEL=anthropic/claude-sonnet-5.5
for M in "$@"; do
  TAG=$(echo "$M" | tr '/.:' '___'); NAME="lsub_b25q4b_red_$TAG"; echo "=== $NAME reduce=$M ($(date -u +%H:%M) UTC)"
  ANSWER_MODEL="$M" RANK_PRIOR_K=0 RANK_PRIOR_K_LIST=10 "$PY" -u mf_answer_r2.py --qa "$QA" --pool runs/san_fused_e_q4b_w15.rank.json --topk 25 --batch 25 --mode mapreduce --workers 6 --out "ans/$NAME.jsonl" 2> "runs/$NAME.err" | tail -1
  echo "answers $(wc -l < "ans/$NAME.jsonl"); FAILED $(grep -c '^FAILED' "runs/$NAME.err"); $(tail -1 "runs/$NAME.err")"
  WORKERS=6 ./judge.sh "$NAME" "$QA" | tee -a runs/list_sub_reducer.txt
done
echo "=== REDUCER PROBE DONE ($(date -u +%H:%M) UTC)"

#!/usr/bin/env bash
# K_LIST sweep on the adopted list configuration (4B-fused pool, one 25-item map call), on the hash subset's 30 list q (deterministic Jaccard).
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
PY="${ATM_PY:-$HOME/eval-archive/atm-run/.venv/bin/python}"; QA="$SP/atm/atm-subset-sha1-list.json"
export ITEMS_FILE="$SP/atm/items-san.json" BEST_GUESS=1 MAP_MODEL=deepseek/deepseek-v4.1-flash FALLBACK_MODEL=anthropic/claude-sonnet-5.5 ANSWER_MODEL=openai/gpt-5.4-mini
for KL in "$@"; do
  NAME="lsub_b25q4b_K$KL"; echo "=== $NAME K_LIST=$KL ($(date +%H:%M))"
  RANK_PRIOR_K=0 RANK_PRIOR_K_LIST=$KL "$PY" -u mf_answer_r2.py --qa "$QA" --pool runs/san_fused_e_q4b_w15.rank.json --topk 25 --batch 25 --mode mapreduce --workers 6 --out "ans/$NAME.jsonl" 2> "runs/$NAME.err" | tail -1
  echo "answers $(wc -l < "ans/$NAME.jsonl"); FAILED $(grep -c '^FAILED' "runs/$NAME.err"); $(tail -1 "runs/$NAME.err")"
  WORKERS=6 ./judge.sh "$NAME" "$QA" | tee -a runs/list_sub_b25q4b.txt
done
echo "=== LIST SUBSET DONE ($(date +%H:%M))"

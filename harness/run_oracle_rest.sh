#!/usr/bin/env bash
# Second half of the subset Oracle sweep (opus-5.5 restarted with 3 workers + stderr captured, then gpt-5.5). Cached calls are reused.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
PY="${ATM_PY:-$HOME/eval-archive/atm-run/.venv/bin/python}"
export ITEMS_FILE="$SP/atm/items-san.json" BEST_GUESS=1 FALLBACK_MODEL=openai/gpt-5.4-mini
for M in anthropic/claude-opus-5.5 openai/gpt-5.5; do
  N="orc_san_$(echo "$M" | tr '/.' '__')"
  [ -s "ans/$N.eval/atm_openai_gpt-5-mini_summary.json" ] && { echo "skip $N"; continue; }
  W=6
  echo "=== $M ($(date +%H:%M)) workers $W"
  ANSWER_MODEL="$M" "$PY" -u mf_answer_v2.py --qa atm/atm-subset-sha1.json --pool gold --mode mapreduce --workers $W --out "ans/$N.jsonl" 2> "runs/$N.err" | grep -v "^$" | tail -2
  echo "stderr lines: $(wc -l < "runs/$N.err") (reduce fallbacks: $(grep -c 'reduce fallback' "runs/$N.err"))"
  ./judge.sh "$N" "$SP/atm/atm-subset-sha1.json" | tee -a runs/oracle_frontier.txt
done
echo "=== ORACLE SWEEP DONE ($(date +%H:%M))"

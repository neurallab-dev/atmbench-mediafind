#!/usr/bin/env bash
# Subset Oracle (gold evidence only, sanitized item text, frozen variant B) for the reduce-model choice; one judge pass each.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
PY="$HOME/atmbench/.venv/bin/python"
export ITEMS_FILE="$SP/atm/items-san.json" BEST_GUESS=1 FALLBACK_MODEL=openai/gpt-5.4-mini
for M in openai/gpt-5.4-mini openai/gpt-5.4 openai/gpt-5.6-sol anthropic/claude-opus-5.5 openai/gpt-5.5; do
  N="orc_san_$(echo "$M" | tr '/.' '__')"
  [ -s "ans/$N.eval/atm_openai_gpt-5-mini_summary.json" ] && { echo "skip $N"; continue; }
  echo "=== $M ($(date +%H:%M))"
  ANSWER_MODEL="$M" "$PY" mf_answer_v2.py --qa atm/atm-subset-sha1.json --pool gold --mode mapreduce --workers 6 --out "ans/$N.jsonl" 2>&1 | grep -v "^$" | tail -2
  ./judge.sh "$N" "$SP/atm/atm-subset-sha1.json" | tee -a runs/oracle_frontier.txt
done
echo "=== ORACLE SWEEP DONE ($(date +%H:%M))"

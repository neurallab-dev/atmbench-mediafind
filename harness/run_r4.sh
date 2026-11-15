#!/usr/bin/env bash
# usage: run_r4.sh sub                      — subset selection stage: both targets x both equality rules, scored from existing judgments
#        run_r4.sh full <target> <equality> — full set: compose answers, one clean judge pass
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; PY="$HOME/atmbench/.venv/bin/python"
export ITEMS_FILE="$SP/atm/items-san.json" BEST_GUESS=1 MAP_MODEL=deepseek/deepseek-v4.1-flash FALLBACK_MODEL=anthropic/claude-sonnet-5.5 ANSWER_MODEL=openai/gpt-5.4-mini
case "$1" in
  sub) echo "=== R4 subset stage ($(date +%H:%M))" | tee -a "$SP/runs/r4_sub.txt"
       for T in r3_full_S2_mini san_full_r2v3; do for E in strict semantic; do
         "$PY" -u "$SP/r4_gate.py" --qa "$SP/atm/atm-subset-sha1.json" --target "$T" --equality "$E" --out "$SP/ans/r4_sub_${T}_$E.jsonl" --score 2>&1 | tee -a "$SP/runs/r4_sub.txt"
       done; done ;;
  full) T="$2"; E="$3"; NAME="r4_full_${T}_$E"; echo "=== R4 full $T $E ($(date +%H:%M))" | tee -a "$SP/runs/r4_full.txt"
       "$PY" -u "$SP/r4_gate.py" --qa "$SP/atm/atm-bench.json" --target "$T" --equality "$E" --out "$SP/ans/$NAME.jsonl" 2>&1 | tee -a "$SP/runs/r4_full.txt"
       WORKERS=6 "$SP/judge.sh" "$NAME" "$SP/atm/atm-bench.json" | tee -a "$SP/runs/r4_full.txt" ;;
esac
echo "=== R4 $1 done ($(date +%H:%M))" | tee -a "$SP/runs/r4_$1.txt"

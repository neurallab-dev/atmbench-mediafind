#!/usr/bin/env bash
# usage: run_r3.sh gate <variant> <SECOND_MODEL> [SECOND_CTX]   — subset gate only (second opinion, deterministic score, no reduce/judge)
#        run_r3.sh sub  <variant> <SECOND_MODEL> [SECOND_CTX]   — subset answers + judge
#        run_r3.sh full <variant> <SECOND_MODEL> [SECOND_CTX]   — full set answers + judge + retrieval line
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; STAGE="$1"; V="$2"; PY="${ATM_PY:-$HOME/eval-archive/atm-run/.venv/bin/python}"
export ITEMS_FILE="$SP/atm/items-san.json" BEST_GUESS=1 MAP_MODEL=deepseek/deepseek-v4.1-flash FALLBACK_MODEL=anthropic/claude-sonnet-5.5 ANSWER_MODEL=openai/gpt-5.4-mini
export SECOND_MODEL="$3" SECOND_CTX="${4:-}" SECOND_K="${SECOND_K:-5}"
case "$STAGE" in
  gate) QA="$SP/atm/atm-subset-sha1.json"; echo "=== R3 gate $V ($(date +%H:%M))" | tee -a "$SP/runs/r3_gate.txt"
        "$PY" -u "$SP/r3_gate.py" --qa "$QA" --pool "$SP/runs/san_fused.rank.json" --topk 25 --workers 6 --out "$SP/runs/r3_gate_$V.json" 2>&1 | tee -a "$SP/runs/r3_gate.txt" ;;
  sub|full) if [ "$STAGE" = sub ]; then QA="$SP/atm/atm-subset-sha1.json"; NAME="r3_sub_$V"; else QA="$SP/atm/atm-bench.json"; NAME="r3_full_$V"; fi
        echo "=== R3 $STAGE $V model=$SECOND_MODEL ctx=$SECOND_CTX K=$SECOND_K ($(date +%H:%M))" | tee -a "$SP/runs/r3_$STAGE.txt"
        "$PY" -u "$SP/mf_answer_r3.py" --qa "$QA" --pool "$SP/runs/san_fused.rank.json" --topk 25 --mode mapreduce --workers 6 --out "$SP/ans/$NAME.jsonl" 2> "$SP/runs/$NAME.err"
        echo "answers written: $(wc -l < "$SP/ans/$NAME.jsonl"); $(tail -1 "$SP/runs/$NAME.err")" | tee -a "$SP/runs/r3_$STAGE.txt"
        WORKERS=6 "$SP/judge.sh" "$NAME" "$QA" | tee -a "$SP/runs/r3_$STAGE.txt" ;;
esac
echo "=== R3 $STAGE $V done ($(date +%H:%M))" | tee -a "$SP/runs/r3_$STAGE.txt"

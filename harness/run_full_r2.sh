#!/usr/bin/env bash
# usage: run_full_r2.sh <name> <K> <K_LIST> — R2 rank prior on all 1,013 questions (sanitized pipeline, frozen variant B,
# gpt-5.4-mini reduce, flash map from cache), one clean gpt-5-mini judge pass, retrieval line (unchanged ranking).
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
NAME="$1"; K="$2"; KL="$3"; QA="$SP/atm/atm-bench.json"; RANK="$SP/runs/san_fused.rank.json"
PY="${ATM_PY:-$HOME/eval-archive/atm-run/.venv/bin/python}"
export ITEMS_FILE="$SP/atm/items-san.json" BEST_GUESS=1 MAP_MODEL=deepseek/deepseek-v4.1-flash FALLBACK_MODEL=anthropic/claude-sonnet-5.5 ANSWER_MODEL=openai/gpt-5.4-mini
echo "=== FULL R2 $NAME K=$K K_LIST=$KL ($(date +%H:%M))"
RANK_PRIOR_K=$K RANK_PRIOR_K_LIST=$KL "$PY" -u mf_answer_r2.py --qa "$QA" --pool "$RANK" --topk 25 --mode mapreduce --workers 6 --out "ans/$NAME.jsonl" 2> "runs/$NAME.err" | tail -1
echo "answers written: $(wc -l < "ans/$NAME.jsonl"); reduce fallbacks $(grep -c 'reduce fallback' "runs/$NAME.err"); FAILED $(grep -c '^FAILED' "runs/$NAME.err"); $(tail -1 "runs/$NAME.err")"
WORKERS=6 ./judge.sh "$NAME" "$QA" "" | tee -a runs/r2_full_result.txt
python3 -I - "$RANK" "$QA" <<'PY'
import json, sys
rank = json.load(open(sys.argv[1])); qa = json.load(open(sys.argv[2]))
def r(k): return 100*sum(sum(e in rank.get(q['question'],[])[:k] for e in q['evidence_ids'])/len(q['evidence_ids']) for q in qa)/len(qa)
print("retrieval: " + "  ".join("R@%d %.2f" % (k, r(k)) for k in (1,5,10,25,50)))
PY
echo "=== FULL R2 $NAME done ($(date +%H:%M))"

#!/usr/bin/env bash
# usage: run_full_lp.sh <name> <rank.json>  — full 1,013-q answer pass on a ranking with the adopted type-conditional policy
# (list questions: reduce also sees the fused top-10; number/open_end: map-kept items only), frozen variant B, flash map,
# gpt-5.4-mini reduce, one gpt-5-mini judge pass, retrieval line.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
NAME="$1"; RANK="$2"; QA="$SP/atm/atm-bench.json"
PY="$HOME/atmbench/.venv/bin/python"
export ITEMS_FILE="$SP/atm/items-san.json" BEST_GUESS=1 MAP_MODEL=deepseek/deepseek-v4.1-flash FALLBACK_MODEL=anthropic/claude-sonnet-5.5 ANSWER_MODEL=openai/gpt-5.4-mini
echo "=== FULL LP $NAME rank=$RANK ($(date +%H:%M))"
RANK_PRIOR_K=0 RANK_PRIOR_K_LIST=10 "$PY" -u mf_answer_r2.py --qa "$QA" --pool "$RANK" --topk 25 --mode mapreduce --workers 6 --out "ans/$NAME.jsonl" 2> "runs/$NAME.err" | tail -1
echo "answers written: $(wc -l < "ans/$NAME.jsonl"); reduce fallbacks $(grep -c 'reduce fallback' "runs/$NAME.err"); FAILED $(grep -c '^FAILED' "runs/$NAME.err"); $(tail -1 "runs/$NAME.err")"
WORKERS=6 ./judge.sh "$NAME" "$QA" "" | tee -a runs/lp_full_result.txt
python3 -I - "$RANK" "$QA" <<'PY'
import json, sys
rank = json.load(open(sys.argv[1])); qa = json.load(open(sys.argv[2]))
def r(k): return 100*sum(sum(e in rank.get(q['question'],[])[:k] for e in q['evidence_ids'])/len(q['evidence_ids']) for q in qa)/len(qa)
print("retrieval: " + "  ".join("R@%d %.2f" % (k, r(k)) for k in (1,5,10,25,50)))
PY
echo "=== FULL LP $NAME done ($(date +%H:%M))"

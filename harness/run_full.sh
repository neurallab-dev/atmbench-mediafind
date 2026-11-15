#!/usr/bin/env bash
# usage: [BEST_GUESS=.. MAP_MODEL=..] run_full.sh <name> <rank.json> <topk> <model> — all 1,013 questions: answers (card items), one clean judge pass, retrieval numbers
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"
NAME="$1"; RANK="$2"; TOPK="$3"; MODEL="$4"; QA="$SP/atm/atm-bench.json"
export ITEMS_FILE="${ITEMS_FILE:-$SP/atm/items-cards.json}"
echo "=== FULL $NAME model=$MODEL rank=$RANK topk=$TOPK BEST_GUESS=${BEST_GUESS:-} MAP_MODEL=${MAP_MODEL:-} ($(date +%H:%M))"
ANSWER_MODEL="$MODEL" "${ATM_PY:-$HOME/eval-archive/atm-run/.venv/bin/python}" "$SP/mf_answer_v2.py" --qa "$QA" --pool "$RANK" --topk "$TOPK" --mode mapreduce --workers "${WORKERS:-6}" --out "$SP/ans/$NAME.jsonl" 2>&1 | grep -v "^$" | tail -5
echo "answers written: $(wc -l < "$SP/ans/$NAME.jsonl"); FAILED lines above (must be 0)"
WORKERS="${JUDGE_WORKERS:-6}" "$SP/judge.sh" "$NAME" "$QA" ""
python3 -I - "$RANK" "$QA" <<'PY'
import json, sys
rank = json.load(open(sys.argv[1])); qa = json.load(open(sys.argv[2]))
def r(k): return 100*sum(sum(e in rank.get(q['question'],[])[:k] for e in q['evidence_ids'])/len(q['evidence_ids']) for q in qa)/len(qa)
print("retrieval: " + "  ".join("R@%d %.2f" % (k, r(k)) for k in (1,5,10,25,50)))
PY
echo "=== FULL $NAME done ($(date +%H:%M))"

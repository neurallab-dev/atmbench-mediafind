#!/usr/bin/env bash
# usage: run_frontier.sh <reduce model> <tag>  — ONE reduce-only pass over the sanitized pipeline (frozen variant B,
# cached flash relevance map, same fused pools), one gpt-5-mini judge pass, evidence with top-10 + top-25 ids.
# stderr of the answerer is kept in runs/san_full_<tag>.err so reduce fallbacks / FAILED questions are countable.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
MODEL="$1"; TAG="$2"; NAME="san_full_$TAG"; QA="$SP/atm/atm-bench.json"; RANK="$SP/runs/san_fused.rank.json"
PY="$HOME/atmbench/.venv/bin/python"
LABEL="$MODEL"; case "$MODEL" in openai/gpt-5*) LABEL="$MODEL (reasoning effort minimal)";; esac
echo "=== frontier reduce-only pass: $MODEL -> $NAME ($(date +%H:%M))"
export ITEMS_FILE="$SP/atm/items-san.json" BEST_GUESS=1 MAP_MODEL=deepseek/deepseek-v4.1-flash FALLBACK_MODEL=openai/gpt-5.4-mini
ANSWER_MODEL="$MODEL" "$PY" -u mf_answer_v2.py --qa "$QA" --pool "$RANK" --topk 25 --mode mapreduce --workers 6 --out "ans/$NAME.jsonl" 2> "runs/$NAME.err" | grep -v "^$" | tail -3
echo "answers written: $(wc -l < "ans/$NAME.jsonl"); stderr lines $(wc -l < "runs/$NAME.err"); reduce fallbacks $(grep -c 'reduce fallback' "runs/$NAME.err"); FAILED $(grep -c '^FAILED' "runs/$NAME.err")"
WORKERS=6 ./judge.sh "$NAME" "$QA" "" | tee -a runs/frontier_result.txt
python3 -I - "$RANK" "$QA" <<'PY'
import json, sys
rank = json.load(open(sys.argv[1])); qa = json.load(open(sys.argv[2]))
def r(k): return 100*sum(sum(e in rank.get(q['question'],[])[:k] for e in q['evidence_ids'])/len(q['evidence_ids']) for q in qa)/len(qa)
print("retrieval: " + "  ".join("R@%d %.2f" % (k, r(k)) for k in (1,5,10,25,50)))
PY
python3 -I make_evidence.py "ans/$NAME.eval" "$RANK" "runs/mediafind-hosted-llms-atmbench-$TAG.json" \
  --answer-model "$LABEL via OpenRouter for the reduce step; deepseek/deepseek-v4.1-flash (reasoning off) via OpenRouter for the relevance map and for rewriting each question into a retrieval query" \
  --memory-model "deepseek/deepseek-v4.1-flash via OpenRouter writes one memory card per photo/video from its caption, the person's emails (a trip/event timeline extracted from all emails) and same-day items; deepseek/deepseek-v4-flash-vision-exp via OpenRouter adds a dense visual description of each photo/video from the raw media" \
  --retriever "BAAI/bge-m3 + BM25 (MediaFind memory search, lexical fusion) + BAAI/bge-reranker-v2-m3 whole-item rerank of the top-50, reciprocal-rank fused with the lexical ranking of the LLM-rewritten query" --topk 25
echo "=== FRONTIER PASS DONE ($(date +%H:%M))"

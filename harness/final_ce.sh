#!/usr/bin/env bash
# Wait for the reranked ranking, then the final full run on its top-25 and the evidence json.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
while [ ! -s runs/cards_full.ce50.rank.json ]; do sleep 60; done
echo "=== reranked ranking ready ($(date +%H:%M))"; grep "^model BAAI" runs/cards_full_ce50b.log | tail -1
python3 -I - <<'PY'
import json
qa = json.load(open("atm/atm-bench.json"))
for name in ("runs/cards_full.lex.rank.json", "runs/cards_full.ce50.rank.json"):
    R = json.load(open(name))
    def r(k): return sum(sum(e in R.get(q["question"], [])[:k] for e in q["evidence_ids"]) / len(q["evidence_ids"]) for q in qa) / len(qa)
    print(name.split("/")[-1], " ".join("R@%d %.3f" % (k, r(k)) for k in (1, 5, 10, 25, 50, 100)))
PY
BEST_GUESS=1 MAP_MODEL=deepseek/deepseek-v4.1-flash FALLBACK_MODEL=anthropic/claude-sonnet-5.5 WORKERS=6 ./run_full.sh full2_cards_ce50_mini runs/cards_full.ce50.rank.json 25 openai/gpt-5.4-mini
python3 -I make_evidence.py ans/full2_cards_ce50_mini.eval runs/cards_full.ce50.rank.json runs/mediafind-atmbench-ce50.json \
  --answer-model "openai/gpt-5.4-mini (reasoning effort minimal) via OpenRouter for the reduce step; deepseek/deepseek-v4.1-flash (reasoning off) via OpenRouter for the relevance map" \
  --memory-model "deepseek/deepseek-v4.1-flash via OpenRouter writes one memory card per photo/video from its caption, the person's emails and same-day items" \
  --retriever "BAAI/bge-m3 + BM25 (MediaFind memory search, lexical fusion) + BAAI/bge-reranker-v2-m3 whole-item rerank of the top-50" --topk 25
echo "=== FINAL DONE ($(date +%H:%M))"

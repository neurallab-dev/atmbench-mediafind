#!/usr/bin/env bash
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
while [ ! -s runs/fused_ce50.rank.json ]; do sleep 60; done
echo "=== gen4 ranking ready ($(date +%H:%M))"; grep "^model BAAI" runs/fused_ce50.log | tail -1
python3 -I - <<'PY'
import json
qa = json.load(open("atm/atm-bench.json"))
F = json.load(open("runs/fused_rw_gen2ce_k10.rank.json")); G = json.load(open("runs/fused_ce50.rank.json"))
def r(R, k): return sum(sum(e in R.get(q["question"], [])[:k] for e in q["evidence_ids"]) / len(q["evidence_ids"]) for q in qa) / len(qa)
for name, R in (("gen3 fused", F), ("gen4 fused+CE", G)):
    print(name, " ".join("R@%d %.3f" % (k, r(R, k)) for k in (1, 5, 10, 25, 50, 100)), "| all gold in top-25:", sum(all(e in R[q["question"]][:25] for e in q["evidence_ids"]) for q in qa))
print("top-25 pools that differ from gen 3:", sum(set(F[q["question"]][:25]) != set(G[q["question"]][:25]) for q in qa))
PY
ITEMS_FILE="$SP/atm/items-cards2.json" BEST_GUESS=1 MAP_MODEL=deepseek/deepseek-v4.1-flash FALLBACK_MODEL=anthropic/claude-sonnet-5.5 WORKERS=6 ./run_full.sh full5_fused_ce50_mini runs/fused_ce50.rank.json 25 openai/gpt-5.4-mini
python3 -I make_evidence.py ans/full5_fused_ce50_mini.eval runs/fused_ce50.rank.json runs/mediafind-atmbench-gen4.json \
  --answer-model "openai/gpt-5.4-mini (reasoning effort minimal) via OpenRouter for the reduce step; deepseek/deepseek-v4.1-flash (reasoning off) via OpenRouter for the relevance map and for rewriting each question into a retrieval query" \
  --memory-model "deepseek/deepseek-v4.1-flash via OpenRouter writes one memory card per photo/video from its caption, the person's emails and same-day items; deepseek/deepseek-v4-flash-vision-exp via OpenRouter adds a dense visual description of each photo/video from the raw media" \
  --retriever "BAAI/bge-m3 + BM25 (MediaFind memory search, lexical fusion) + BAAI/bge-reranker-v2-m3 whole-item rerank of the top-50, reciprocal-rank fused with the lexical ranking of the LLM-rewritten query, then reranked again over the fused top-50" --topk 25
echo "=== GEN4 DONE ($(date +%H:%M))"

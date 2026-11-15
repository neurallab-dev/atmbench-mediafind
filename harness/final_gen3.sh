#!/usr/bin/env bash
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
echo "=== gen3: fused(rewritten-query lexical, gen2 whole-item rerank) top-25 ($(date +%H:%M))"
ITEMS_FILE="$SP/atm/items-cards2.json" BEST_GUESS=1 MAP_MODEL=deepseek/deepseek-v4.1-flash FALLBACK_MODEL=anthropic/claude-sonnet-5.5 WORKERS=4 ./run_full.sh full4_fused_rw_mini runs/fused_rw_gen2ce_k10.rank.json 25 openai/gpt-5.4-mini
python3 -I make_evidence.py ans/full4_fused_rw_mini.eval runs/fused_rw_gen2ce_k10.rank.json runs/mediafind-atmbench-gen3.json \
  --answer-model "openai/gpt-5.4-mini (reasoning effort minimal) via OpenRouter for the reduce step; deepseek/deepseek-v4.1-flash (reasoning off) via OpenRouter for the relevance map and for rewriting each question into a retrieval query" \
  --memory-model "deepseek/deepseek-v4.1-flash via OpenRouter writes one memory card per photo/video from its caption, the person's emails and same-day items; deepseek/deepseek-v4-flash-vision-exp via OpenRouter adds a dense visual description of each photo/video from the raw media" \
  --retriever "BAAI/bge-m3 + BM25 (MediaFind memory search, lexical fusion) + BAAI/bge-reranker-v2-m3 whole-item rerank of the top-50, reciprocal-rank fused with the lexical ranking of the LLM-rewritten query" --topk 25
echo "=== GEN3 DONE ($(date +%H:%M))"

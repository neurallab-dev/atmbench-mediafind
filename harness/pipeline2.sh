#!/usr/bin/env bash
# Second generation: cards + dense visual descriptions -> tasks -> CPU ingest + lexical -> whole-item rerank (after the
# first reranker has finished, one Metal process at a time) -> full answer run -> judge -> evidence.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
PY="${ATM_PY:-$HOME/eval-archive/atm-run/.venv/bin/python}"
stage() { echo "=== $1 ($(date +%H:%M))"; }
while ! grep -q "^done" runs/vision.log 2>/dev/null; do sleep 60; done
WORKERS=16 "$PY" -u vlm_caption.py cards/vision.jsonl deepseek/deepseek-v4-flash-vision-exp >> runs/vision.log 2>&1   # retry pass for failures
echo "vision descriptions: $(wc -l < cards/vision.jsonl)"
[ -s atm/atm-bench-cards2.doc.jsonl ] || { stage "tasks (cards + vision)"; python3 -I build_tasks.py cards/cards.jsonl cards2 head cards/vision.jsonl; }
if [ ! -s runs/cards2_full.lex.rank.json ]; then
  stage "ingest + lexical ranking (cards2)"
  rm -rf data_cards2
  MEDIAFIND_EMBED_DEVICE=cpu ./run_rank.sh data_cards2 atm/atm-bench-cards2.doc.jsonl runs/cards2_full lex:lexical > runs/cards2_full.log 2>&1; grep -E "RESULT|ingest\] done" runs/cards2_full.log
fi
while [ ! -s runs/cards_full.ce50.rank.json ]; do sleep 60; done   # one Metal reranker at a time
if [ ! -s runs/cards2_full.ce50.rank.json ]; then
  stage "whole-item rerank top-50 (cards2)"
  export HF_HOME="$HOME/Library/Application Support/MediaFind/models/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  ITEMS_FILE="$SP/atm/items-cards2.json" CE_DEVICE=mps CE_BS=4 /Users/mbp/code/mediafind/.venv/bin/python -u ce_screen.py runs/cards2_full.lex.rank.json runs/cards2_full.ce50.rank.json BAAI/bge-reranker-v2-m3 50 1024 > runs/cards2_full_ce50.log 2>&1; tail -1 runs/cards2_full_ce50.log
fi
python3 -I - <<'PY'
import json
qa = json.load(open("atm/atm-bench.json"))
for name in ("runs/cards_full.lex.rank.json", "runs/cards_full.ce50.rank.json", "runs/cards2_full.lex.rank.json", "runs/cards2_full.ce50.rank.json"):
    try: R = json.load(open(name))
    except Exception: continue
    def r(k): return sum(sum(e in R.get(q["question"], [])[:k] for e in q["evidence_ids"]) / len(q["evidence_ids"]) for q in qa) / len(qa)
    print(name.split("/")[-1], " ".join("R@%d %.3f" % (k, r(k)) for k in (1, 5, 10, 25, 50, 100)))
PY
stage "full answer run + judge (cards2, reranked)"
ITEMS_FILE="$SP/atm/items-cards2.json" BEST_GUESS=1 MAP_MODEL=deepseek/deepseek-v4.1-flash FALLBACK_MODEL=anthropic/claude-sonnet-5.5 WORKERS=6 ./run_full.sh full3_cards2_ce50_mini runs/cards2_full.ce50.rank.json 25 openai/gpt-5.4-mini
python3 -I make_evidence.py ans/full3_cards2_ce50_mini.eval runs/cards2_full.ce50.rank.json runs/mediafind-atmbench-cards2-ce50.json \
  --answer-model "openai/gpt-5.4-mini (reasoning effort minimal) via OpenRouter for the reduce step; deepseek/deepseek-v4.1-flash (reasoning off) via OpenRouter for the relevance map" \
  --memory-model "deepseek/deepseek-v4.1-flash via OpenRouter writes one memory card per photo/video from its caption, the person's emails and same-day items; deepseek/deepseek-v4-flash-vision-exp via OpenRouter adds a dense visual description of each photo/video from the raw media" \
  --retriever "BAAI/bge-m3 + BM25 (MediaFind memory search, lexical fusion) + BAAI/bge-reranker-v2-m3 whole-item rerank of the top-50" --topk 25
stage "PIPELINE2 DONE"

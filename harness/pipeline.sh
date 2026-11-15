#!/usr/bin/env bash
# End-to-end rebuild after the /private/tmp wipe: trips -> cards -> tasks -> bge-m3 ingest + lexical ranking (full set)
# -> offline whole-item rerank top-50 -> full 1,013-q map-reduce answers -> gpt-5-mini judge -> evidence json.
# Resumable: every stage skips when its output exists.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
PY="$HOME/atmbench/.venv/bin/python"
stage() { echo "=== $1 ($(date +%H:%M))"; }
[ -s atm/items.json ] || { stage "items"; python3 -I make_items.py; }
if [ ! -s cards/trips.json ]; then stage "trips"; WORKERS=12 "$PY" -u trips_extract.py cards/trips.json > runs/trips.log 2>&1; tail -1 runs/trips.log; fi
if [ "$(wc -l < cards/cards.jsonl 2>/dev/null || echo 0)" -lt 4292 ]; then
  stage "cards"; WORKERS=24 "$PY" -u cards.py cards/cards.jsonl deepseek/deepseek-v4.1-flash > runs/cards.log 2>&1; tail -1 runs/cards.log
  WORKERS=24 "$PY" -u cards.py cards/cards.jsonl deepseek/deepseek-v4.1-flash >> runs/cards.log 2>&1   # second pass picks up any failures
fi
echo "cards: $(wc -l < cards/cards.jsonl)"
[ -s atm/atm-bench-cards.doc.jsonl ] || { stage "tasks"; python3 -I build_tasks.py cards/cards.jsonl cards head; }
if [ ! -s runs/cards_full.lex.rank.json ]; then
  stage "ingest + lexical ranking (full set)"
  rm -rf data_cards   # a partial ingest has no marker; start clean
  MEDIAFIND_EMBED_DEVICE=cpu ./run_rank.sh data_cards atm/atm-bench-cards.doc.jsonl runs/cards_full lex:lexical > runs/cards_full.log 2>&1; grep -E "RESULT|ingest\] done" runs/cards_full.log
fi
if [ ! -s runs/cards_full.ce50.rank.json ]; then
  stage "offline whole-item rerank top-50"
  export HF_HOME="$HOME/Library/Application Support/MediaFind/models/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  ITEMS_FILE="$SP/atm/items-cards.json" CE_DEVICE="${CE_DEVICE:-mps}" CE_BS="${CE_BS:-4}" /Users/mbp/code/mediafind/.venv/bin/python -u ce_screen.py runs/cards_full.lex.rank.json runs/cards_full.ce50.rank.json BAAI/bge-reranker-v2-m3 50 1024 > runs/cards_full_ce50.log 2>&1; tail -1 runs/cards_full_ce50.log
fi
python3 -I - <<'PY'
import json
qa = json.load(open("atm/atm-bench.json"))
for name in ("runs/cards_full.lex.rank.json", "runs/cards_full.ce50.rank.json"):
    R = json.load(open(name))
    def r(k): return sum(sum(e in R.get(q["question"], [])[:k] for e in q["evidence_ids"]) / len(q["evidence_ids"]) for q in qa) / len(qa)
    print(name.split("/")[-1], " ".join("R@%d %.3f" % (k, r(k)) for k in (1, 5, 10, 25, 50, 100)))
PY
if [ ! -s ans/full1_cards_ce50_mini.eval/atm_openai_gpt-5-mini_summary.json ]; then
  stage "full answer run + judge"
  BEST_GUESS=1 MAP_MODEL=deepseek/deepseek-v4.1-flash FALLBACK_MODEL=anthropic/claude-sonnet-5.5 WORKERS=6 ./run_full.sh full1_cards_ce50_mini runs/cards_full.ce50.rank.json 25 openai/gpt-5.4-mini
fi
stage "evidence"
python3 -I make_evidence.py ans/full1_cards_ce50_mini.eval runs/cards_full.ce50.rank.json runs/mediafind-atmbench.json \
  --answer-model "openai/gpt-5.4-mini (reasoning effort minimal) via OpenRouter for the reduce step; deepseek/deepseek-v4.1-flash (reasoning off) via OpenRouter for the relevance map" \
  --memory-model "deepseek/deepseek-v4.1-flash via OpenRouter writes one memory card per photo/video from its caption, the person's emails and same-day items" \
  --retriever "BAAI/bge-m3 + BM25 (MediaFind memory search, lexical fusion) + BAAI/bge-reranker-v2-m3 whole-item rerank of the top-50" --topk 25
stage "ALL DONE"

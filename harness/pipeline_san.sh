#!/usr/bin/env bash
# Sanitized re-run (2026-10-23): timeline -> cards -> vision -> tasks -> CPU ingest + lexical -> whole-item rerank (MPS)
# -> rewrites (facts from the new timeline) -> rewritten-query lexical -> RRF k=10 -> variant choice on the sha1 subset
# -> full run with the frozen variant -> judge -> evidence (top-10 + top-25 ids). Resumable: stages skip when their output exists.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
PY="$HOME/atmbench/.venv/bin/python"
stage() { echo "=== $1 ($(date +%H:%M))"; }
export HF_HOME="$HOME/Library/Application Support/MediaFind/models/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [ ! -s cards/trips_s.json ]; then stage "timeline"; WORKERS=12 "$PY" -u trips_extract.py cards/trips_s.json > runs/trips_s.log 2>&1; tail -1 runs/trips_s.log; fi
if [ "$(wc -l < cards/cards_s.jsonl 2>/dev/null || echo 0)" -lt 4280 ]; then
  stage "cards"; TRIPS="$SP/cards/trips_s.json" WORKERS=24 "$PY" -u cards.py cards/cards_s.jsonl deepseek/deepseek-v4.1-flash > runs/cards_s.log 2>&1; tail -1 runs/cards_s.log
  TRIPS="$SP/cards/trips_s.json" WORKERS=24 "$PY" -u cards.py cards/cards_s.jsonl deepseek/deepseek-v4.1-flash >> runs/cards_s.log 2>&1
fi
echo "cards: $(wc -l < cards/cards_s.jsonl)"
if [ "$(wc -l < cards/vision_s.jsonl 2>/dev/null || echo 0)" -lt 4280 ]; then
  stage "vision"; WORKERS=16 "$PY" -u vlm_caption.py cards/vision_s.jsonl deepseek/deepseek-v4-flash-vision-exp > runs/vision_s.log 2>&1; tail -1 runs/vision_s.log
  WORKERS=16 "$PY" -u vlm_caption.py cards/vision_s.jsonl deepseek/deepseek-v4-flash-vision-exp >> runs/vision_s.log 2>&1
fi
echo "vision: $(wc -l < cards/vision_s.jsonl)"
[ -s atm/atm-bench-san.doc.jsonl ] || { stage "tasks"; python3 -I build_tasks.py cards/cards_s.jsonl san head cards/vision_s.jsonl; }
if [ ! -s runs/san_full.lex.rank.json ]; then
  stage "ingest + lexical (CPU)"; rm -rf data_san
  MEDIAFIND_EMBED_DEVICE=cpu ./run_rank.sh data_san atm/atm-bench-san.doc.jsonl runs/san_full lex:lexical > runs/san_full.log 2>&1; grep -E "RESULT|ingest\] done" runs/san_full.log
fi
[ -s runs/san_full.lex.rank.json ] || { echo "ABORT: lexical ranking missing (see runs/san_full.log)"; exit 1; }
if [ ! -s runs/san_full.ce50.rank.json ]; then
  stage "whole-item rerank top-50 (MPS)"
  ITEMS_FILE="$SP/atm/items-san.json" CE_DEVICE=mps CE_BS=4 /Users/mbp/code/mediafind/.venv/bin/python -u ce_screen.py runs/san_full.lex.rank.json runs/san_full.ce50.rank.json BAAI/bge-reranker-v2-m3 50 1024 > runs/san_full_ce50.log 2>&1; tail -1 runs/san_full_ce50.log
fi
if [ ! -s runs/rw_san.lex.rank.json ]; then
  stage "question rewrites + rewritten-query lexical"
  TRIPS="$SP/cards/trips_s.json" TASKS_TAG=san "$PY" -u rewrite_queries.py runs/rewrites_s.json > runs/rewrite_s.log 2>&1; head -3 runs/rewrite_s.log
  MEDIAFIND_EMBED_DEVICE=cpu ./run_rank.sh data_san atm/atm-bench-san-rw.doc.jsonl runs/rw_san lex:lexical > runs/rw_san.log 2>&1; grep RESULT runs/rw_san.log
fi
[ -s runs/rw_san.lex.rank.json ] && [ -s runs/san_full.ce50.rank.json ] || { echo "ABORT: a ranking is missing"; exit 1; }
stage "fusion"
python3 -I - <<'PY'
import json
qa = json.load(open("atm/atm-bench.json")); byid = {q["id"]: q for q in qa}
key = json.load(open("atm/rw-san-key.json")); RW = json.load(open("runs/rw_san.lex.rank.json"))
RWq = {byid[key[k]]["question"]: v for k, v in RW.items() if k in key}
B = json.load(open("runs/san_full.ce50.rank.json")); BL = json.load(open("runs/san_full.lex.rank.json"))
def recall(R, k): return sum(sum(e in R.get(q["question"], [])[:k] for e in q["evidence_ids"]) / len(q["evidence_ids"]) for q in qa) / len(qa)
def rrf(*Rs, k=10, depth=100):
    out = {}
    for q in qa:
        sc = {}
        for R in Rs:
            for r, i in enumerate(R.get(q["question"], [])[:depth]): sc[i] = sc.get(i, 0) + 1 / (k + r + 1)
        out[q["question"]] = [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])]
    return out
F = rrf(RWq, B)
for name, R in (("san lexical", BL), ("san reranked", B), ("san rewritten-query lexical", RWq), ("san fused (submitted ranking)", F)):
    print(f"{name:32s} " + " ".join(f"R@{k} {recall(R, k):.3f}" for k in (1, 5, 10, 25, 50, 100)))
json.dump(F, open("runs/san_fused.rank.json", "w"))
PY
stage "variant choice on the sha1 subset"
export ITEMS_FILE="$SP/atm/items-san.json"
for V in A B; do
  if [ ! -s "ans/san_sub_$V.eval/atm_openai_gpt-5-mini_summary.json" ]; then
    if [ "$V" = B ]; then export BEST_GUESS=1; else unset BEST_GUESS; fi
    MAP_MODEL=deepseek/deepseek-v4.1-flash ANSWER_MODEL=openai/gpt-5.4-mini FALLBACK_MODEL=anthropic/claude-sonnet-5.5 "$PY" mf_answer_v2.py --qa atm/atm-subset-sha1.json --pool runs/san_fused.rank.json --topk 25 --workers 6 --out "ans/san_sub_$V.jsonl" 2>&1 | tail -1
    ./judge.sh "san_sub_$V" "$SP/atm/atm-subset-sha1.json" | tee -a runs/variant_choice.txt
  fi
done
unset BEST_GUESS
CHOICE=$(python3 -c "
import json
a = json.load(open('ans/san_sub_A.eval/atm_openai_gpt-5-mini_summary.json'))['accuracy']; b = json.load(open('ans/san_sub_B.eval/atm_openai_gpt-5-mini_summary.json'))['accuracy']
print('B' if b > a else 'A'); open('runs/variant_choice.txt', 'a').write(f'subset QS: A {100*a:.2f}  B {100*b:.2f}  -> frozen variant {\"B\" if b > a else \"A\"}\n')")
[ -n "$CHOICE" ] || { echo "ABORT: no variant chosen"; exit 1; }
echo "frozen variant: $CHOICE"; cat runs/variant_choice.txt
if [ ! -s ans/san_full.eval/atm_openai_gpt-5-mini_summary.json ]; then
  stage "full run with the frozen variant ($CHOICE)"
  if [ "$CHOICE" = B ]; then export BEST_GUESS=1; fi
  MAP_MODEL=deepseek/deepseek-v4.1-flash FALLBACK_MODEL=anthropic/claude-sonnet-5.5 WORKERS=6 ./run_full.sh san_full runs/san_fused.rank.json 25 openai/gpt-5.4-mini
fi
stage "evidence"
python3 -I make_evidence.py ans/san_full.eval runs/san_fused.rank.json runs/mediafind-v2-atmbench-san.json \
  --answer-model "openai/gpt-5.4-mini (reasoning effort minimal) via OpenRouter for the reduce step; deepseek/deepseek-v4.1-flash (reasoning off) via OpenRouter for the relevance map and for rewriting each question into a retrieval query" \
  --memory-model "deepseek/deepseek-v4.1-flash via OpenRouter writes one memory card per photo/video from its caption, the person's emails (a trip/event timeline extracted from all emails) and same-day items; deepseek/deepseek-v4-flash-vision-exp via OpenRouter adds a dense visual description of each photo/video from the raw media" \
  --retriever "BAAI/bge-m3 + BM25 (MediaFind memory search, lexical fusion) + BAAI/bge-reranker-v2-m3 whole-item rerank of the top-50, reciprocal-rank fused with the lexical ranking of the LLM-rewritten query" --topk 25
stage "SANITIZED RUN DONE"

#!/usr/bin/env bash
# ONE end-to-end re-run of the frozen "MediaFind + hosted LLMs" configuration (the QS 80.69 row of 2026-10-29), from the
# released dataset to one judged prediction file. Resumable: every stage skips when its output exists. Paid stages first
# check the OpenRouter balance and stop (exit 3) when it is below the stage's estimate plus MARGIN. The judge runs once,
# on the assembled 1,013-answer file, and nothing is re-run after it.
# usage: rerun_frozen.sh      env: MF_PIN_BGE (bge-m3 retrieval tree, default ./mf-6c783997), MF_PIN_4B (./mf-bfacc365),
#                                 MEDIA (raw photos/videos: <MEDIA>/image/<id>.jpg, <MEDIA>/video/<id>.mp4), MARGIN (default 0.5)
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
PY="${ATM_PY:-$SP/.venv/bin/python}"; PYV="${MF_PY:-/Users/mbp/code/mediafind/.venv/bin/python}"
PIN_BGE="${MF_PIN_BGE:-$SP/mf-6c783997}"; PIN_4B="${MF_PIN_4B:-$SP/mf-bfacc365}"; MEDIA="${MEDIA:-$SP/media}"; MARGIN="${MARGIN:-0.5}"
export HF_HOME="$HOME/Library/Application Support/MediaFind/models/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p atm cards runs ans
stage() { echo "=== $1 ($(date -u +%Y-%m-%dT%H:%MZ))"; }
need() {  # need <estimated $> <stage name>
  local rem; rem=$("$PY" -I - <<'PY'
import json, os, re, urllib.request
p = os.path.expanduser(os.environ.get("OPENROUTER_ENV_FILE", "~/.config/m3bench/openrouter.env"))
k = next(re.match(r"\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(.+)", l).group(1).strip().strip("\"'") for l in open(p) if "OPENROUTER_API_KEY" in l)
d = json.load(urllib.request.urlopen(urllib.request.Request("https://openrouter.ai/api/v1/auth/key", headers={"Authorization": "Bearer " + k}), timeout=30))["data"]
print(d.get("limit_remaining"))
PY
)
  echo "    balance \$$rem before '$2' (estimate \$$1, margin \$$MARGIN)"
  if "$PY" -c "import sys; sys.exit(0 if float('$rem') >= $1 + $MARGIN else 1)"; then return 0; fi
  echo "STOP: OpenRouter balance too low for '$2'. Top up and run again; finished stages are kept."; exit 3
}
lines() { [ -s "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
answers_ok() { local f; f=$(grep -c '^FAILED' "$3" 2>/dev/null); [ "$(lines "$1")" -eq "$2" ] && [ "${f:-0}" -eq 0 ]; }

# 1. dataset -> official item texts, questions, per-type question files (local)
if [ ! -s atm/items.json ]; then stage "items"; "$PY" -I make_items.py && "$PY" -I make_qa_types.py || exit 1; fi
# 2-4. memory text (paid): timeline, cards, visual descriptions; cards and descriptions run twice to fill provider gaps
if [ ! -s cards/trips_s.json ]; then need 0.5 timeline; stage "timeline"; WORKERS=12 "$PY" -u trips_extract.py cards/trips_s.json > runs/trips_s.log 2>&1; tail -1 runs/trips_s.log; fi
[ -s cards/trips_s.json ] || { echo "ABORT: timeline missing"; exit 1; }
if [ "$(lines cards/cards_s.jsonl)" -lt 4280 ]; then
  need 2.0 cards; stage "cards"
  for pass in 1 2; do TRIPS="$SP/cards/trips_s.json" WORKERS=24 "$PY" -u cards.py cards/cards_s.jsonl deepseek/deepseek-v4.1-flash >> runs/cards_s.log 2>&1; done; tail -1 runs/cards_s.log
fi
echo "cards: $(lines cards/cards_s.jsonl)"
if [ "$(lines cards/vision_s.jsonl)" -lt 4280 ]; then
  need 2.0 vision; stage "vision"
  for pass in 1 2; do MEDIA="$MEDIA" WORKERS=16 "$PY" -u vlm_caption.py cards/vision_s.jsonl deepseek/deepseek-v4-flash-vision-exp >> runs/vision_s.log 2>&1; done; tail -1 runs/vision_s.log
fi
echo "vision: $(lines cards/vision_s.jsonl)"
[ "$(lines cards/cards_s.jsonl)" -ge 4280 ] && [ "$(lines cards/vision_s.jsonl)" -ge 4280 ] || { echo "ABORT: cards or visual descriptions incomplete"; exit 1; }
[ -s atm/atm-bench-san.doc.jsonl ] || { stage "tasks"; "$PY" -I build_tasks.py cards/cards_s.jsonl san head cards/vision_s.jsonl || exit 1; }
# 5. MediaFind retrieval with bge-m3 (local, pinned tree), whole-item rerank of the top-50 (local)
if [ ! -s runs/san_full.lex.rank.json ]; then
  stage "ingest + lexical ranking (bge-m3, CPU)"; rm -rf data_san
  MF_WORKTREE="$PIN_BGE" MEDIAFIND_EMBED_DEVICE=cpu ./run_rank.sh data_san atm/atm-bench-san.doc.jsonl runs/san_full lex:lexical > runs/san_full.log 2>&1; grep -E "RESULT" runs/san_full.log
fi
[ -s runs/san_full.lex.rank.json ] || { echo "ABORT: lexical ranking missing (runs/san_full.log)"; exit 1; }
if [ ! -s runs/san_full.ce50.rank.json ]; then
  stage "whole-item rerank top-50 (MPS)"
  ITEMS_FILE="$SP/atm/items-san.json" CE_DEVICE=mps CE_BS=4 "$PYV" -u ce_screen.py runs/san_full.lex.rank.json runs/san_full.ce50.rank.json BAAI/bge-reranker-v2-m3 50 1024 > runs/san_full_ce50.log 2>&1; tail -1 runs/san_full_ce50.log
fi
[ -s runs/san_full.ce50.rank.json ] || { echo "ABORT: rerank missing"; exit 1; }
# 6. question rewrites (paid) and the bge-m3 ranking of the DeepSeek rewrite (local) -> open-ended pool
if [ ! -s atm/rw-san-key.json ]; then need 0.3 "rewrites (DeepSeek)"; stage "rewrites DeepSeek V4.1 Flash"; TRIPS="$SP/cards/trips_s.json" TASKS_TAG=san "$PY" -u rewrite_queries.py runs/rewrites_s.json > runs/rewrite_s.log 2>&1; head -1 runs/rewrite_s.log; fi
if [ ! -s runs/rw_san.lex.rank.json ]; then stage "rewritten-query lexical ranking (bge-m3)"; MF_WORKTREE="$PIN_BGE" MEDIAFIND_EMBED_DEVICE=cpu ./run_rank.sh data_san atm/atm-bench-san-rw.doc.jsonl runs/rw_san lex:lexical > runs/rw_san.log 2>&1; grep RESULT runs/rw_san.log; fi
[ -s runs/san_fused.rank.json ] || { stage "open-ended pool"; "$PY" -I make_pool_san.py || exit 1; }
rewrite_set() {  # rewrite_set TAG MODEL ESTIMATE
  if [ ! -s "atm/rw-$1-key.json" ]; then
    need "$3" "rewrites ($2)"; stage "rewrites $2"; cp atm/atm-bench-san.doc.jsonl "atm/atm-bench-$1.doc.jsonl"
    TRIPS="$SP/cards/trips_s.json" TASKS_TAG="$1" "$PY" -u rewrite_queries.py "runs/rewrites_$1.json" "$2" > "runs/rewrite_$1.log" 2>&1; head -1 "runs/rewrite_$1.log"
  fi
  [ -s "atm/rw-$1-key.json" ] || { echo "ABORT: rewrites $1 missing"; exit 1; }
}
rewrite_set sanmini openai/gpt-5.4-mini 0.8
# 7. Qwen3-Embedding-4B ranking (local): document vectors over the store's own chunks, injected into a copy of the store,
#    product retrieval (pinned tree) with the model's query prompt, top-50 rerank, fusion -> list and number pool
if [ ! -s runs/pe_q4b_units.npy ]; then stage "Qwen3-Embedding-4B encode (MPS fp16)"; PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 "$PYV" proxy_embed.py Qwen/Qwen3-Embedding-4B q4b --dtype fp16 --maxlen 1024 --devices mps:8,mps:4,mps:2 > runs/pe_q4b.log 2>&1; tail -3 runs/pe_q4b.log; fi
[ -s runs/pe_q4b_units.npy ] || { echo "ABORT: 4B vectors missing"; exit 1; }
if [ ! -s runs/e_q4b.ce50.rank.json ] || [ ! -s runs/e_q4b_rw_sanmini.lex.rank.json ]; then
  stage "4B retrieval (question, DeepSeek and GPT-5.4 mini rewrites) + rerank"; MF_WORKTREE="$PIN_4B" S2_DEVICE=mps MF_QUERY_PROMPT=1 ./run_s2_embed.sh Qwen/Qwen3-Embedding-4B q4b san sanmini > runs/s2_q4b.log 2>&1; tail -4 runs/s2_q4b.log
fi
[ -s runs/e_q4b.ce50.rank.json ] || { echo "ABORT: 4B rerank missing (runs/s2_q4b.log)"; exit 1; }
rewrite_set sangpt54 openai/gpt-5.4 5.0
if [ ! -s runs/e_q4b_rw_sangpt54.lex.rank.json ]; then
  stage "4B retrieval (GPT-5.4 rewrite)"; MF_WORKTREE="$PIN_4B" S2_DEVICE=mps MF_QUERY_PROMPT=1 ./run_s2_embed.sh Qwen/Qwen3-Embedding-4B q4b san sanmini sangpt54 >> runs/s2_q4b.log 2>&1; tail -2 runs/s2_q4b.log
fi
[ -s runs/e_q4b_rw_sangpt54.lex.rank.json ] || { echo "ABORT: 4B ranking of the GPT-5.4 rewrite missing"; exit 1; }
[ -s runs/san_fused_e_q4b_w15.rank.json ] || { stage "list and number pool"; "$PY" -I make_pool_w15.py || exit 1; }
[ -s runs/san_fused_typecond2.rank.json ] || "$PY" -I make_pool_typecond2.py
# 8. answers (paid), one configuration per question type, exactly as the adopted runs; no judge here
export ITEMS_FILE="$SP/atm/items-san.json" BEST_GUESS=1 FALLBACK_MODEL=anthropic/claude-sonnet-5.5 ANSWER_MODEL=openai/gpt-5.4-mini
if ! answers_ok ans/rr_open.jsonl 514 runs/rr_open.err; then
  need 3.0 "open-ended answers"; stage "answers: open_end (514)"
  MAP_MODEL=deepseek/deepseek-v4.1-flash "$PY" -u mf_answer_v2.py --qa atm/atm-bench-open.json --pool runs/san_fused.rank.json --topk 25 --batch 10 --mode mapreduce --workers 6 --out ans/rr_open.jsonl 2> runs/rr_open.err | tail -1
fi
if ! answers_ok ans/rr_list.jsonl 139 runs/rr_list.err; then
  need 1.0 "list answers"; stage "answers: list_recall (139)"
  MAP_MODEL=deepseek/deepseek-v4.1-flash RANK_PRIOR_K=0 RANK_PRIOR_K_LIST=10 "$PY" -u mf_answer_r2.py --qa atm/atm-bench-list-only.json --pool runs/san_fused_e_q4b_w15.rank.json --topk 25 --batch 25 --mode mapreduce --workers 6 --out ans/rr_list.jsonl 2> runs/rr_list.err | tail -1
fi
if ! answers_ok ans/rr_number.jsonl 360 runs/rr_number.err; then
  need 2.5 "number answers"; stage "answers: number (360)"
  MAP_MODEL=anthropic/claude-haiku-5.5 RANK_PRIOR_K=0 RANK_PRIOR_K_LIST=10 "$PY" -u mf_answer_r2.py --qa atm/atm-bench-number.json --pool runs/san_fused_e_q4b_w15.rank.json --topk 25 --batch 10 --mode mapreduce --workers 6 --out ans/rr_number.jsonl 2> runs/rr_number.err | tail -1
fi
for t in open:514 list:139 number:360; do n=${t%%:*}; c=${t##*:}; answers_ok "ans/rr_$n.jsonl" "$c" "runs/rr_$n.err" || { echo "ABORT: ans/rr_$n.jsonl incomplete or has FAILED lines (re-run this script: cached calls are free)"; exit 1; }; done
# 9. one prediction file, one judge pass (official evaluator: em for number/list, gpt-5-mini for open_end)
FINAL=mediafind-hosted-llms-rerun
if [ ! -s "ans/$FINAL.eval/atm_openai_gpt-5-mini_summary.json" ]; then
  "$PY" -I assemble.py "ans/$FINAL.jsonl" ans/rr_number.jsonl ans/rr_list.jsonl ans/rr_open.jsonl | tee runs/final_sha256.txt || exit 1
  need 1.5 judge; stage "judge (once)"; WORKERS=6 ./judge.sh "$FINAL" "$SP/atm/atm-bench.json" "" | tee runs/final_result.txt
fi
stage "RE-RUN DONE"; cat runs/final_result.txt runs/final_sha256.txt

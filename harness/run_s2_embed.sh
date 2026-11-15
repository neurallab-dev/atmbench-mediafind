#!/usr/bin/env bash
# usage: run_s2_embed.sh MODEL TAG [RWTAG...]  (default rewrite tags: san sanmini)
# Product retrieval with a candidate embedder: proxy vectors injected into a copy of the sanitized store, MediaFind lexical mode
# k=100 for the original and the rewritten questions (queries embedded by the product with MEDIAFIND_EMBED=MODEL), whole-item
# CE50 (bge-reranker-v2-m3) on the new original list, RRF k=10 d100 fusions, gate vs the submitted ranking (e_gate.py).
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
MODEL="$1"; TAG="$2"; shift 2; RWT=("${@:-san}"); [ $# -eq 0 ] && RWT=(san sanmini)
PYV=/Users/mbp/code/mediafind/.venv/bin/python
echo "=== S2 $TAG $MODEL rewrites: ${RWT[*]} ($(date +%H:%M))"
[ -s "data_$TAG/ingested-atm.json" ] || $PYV inject_vectors.py "$MODEL" "$TAG" || exit 1
if [ ! -s "runs/e_$TAG.lex.rank.json" ]; then
  MEDIAFIND_EMBED="$MODEL" MEDIAFIND_EMBED_DEVICE="${S2_DEVICE:-cpu}" ./run_rank.sh "data_$TAG" atm/atm-bench-san.doc.jsonl "runs/e_$TAG" lex:lexical > "runs/e_$TAG.log" 2>&1
  grep -E 'RESULT|backend|reusing' "runs/e_$TAG.log"
fi
for t in "${RWT[@]}"; do
  if [ ! -s "runs/e_${TAG}_rw_$t.lex.rank.json" ]; then
    MEDIAFIND_EMBED="$MODEL" MEDIAFIND_EMBED_DEVICE="${S2_DEVICE:-cpu}" ./run_rank.sh "data_$TAG" "atm/atm-bench-$t-rw.doc.jsonl" "runs/e_${TAG}_rw_$t" lex:lexical > "runs/e_${TAG}_rw_$t.log" 2>&1
    grep RESULT "runs/e_${TAG}_rw_$t.log"
  fi
done
if [ ! -s "runs/e_$TAG.ce50.rank.json" ]; then
  echo "=== CE50 ($(date +%H:%M))"
  ITEMS_FILE="$SP/atm/items-san.json" CE_DEVICE="${CE_DEVICE:-mps}" CE_BS="${CE_BS:-4}" $PYV -u ce_screen.py "runs/e_$TAG.lex.rank.json" "runs/e_$TAG.ce50.rank.json" BAAI/bge-reranker-v2-m3 50 1024 > "runs/e_${TAG}_ce50.log" 2>&1; tail -1 "runs/e_${TAG}_ce50.log"
fi
$PYV e_gate.py "$TAG" "${RWT[@]}" | tee "runs/e_${TAG}_gate.txt"
echo "=== S2 $TAG done ($(date +%H:%M))"

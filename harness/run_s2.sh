#!/usr/bin/env bash
# Chunking lever, step 2 (local, $0): re-ingest with the proxy's winning chunking, rank original + rewritten queries (k=100),
# cross-encoder over the new original-query top-50, RRF k=10 depth 100 of (rw100, ce50) = the submitted pipeline's shape, gate.
# usage: run_s2.sh <B|C|D>     B = whole item (MF_DOC_CHUNK_WORDS=2000), C = 160-word chunks + header prefix, D = 400-word chunks
# Pre-stated gate: uncovered@25 <= 71 (now 86) AND R@10 >= 89.43 (now 89.73, -0.3); newly covered / lost coverage reported.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"; PY="$HOME/atmbench/.venv/bin/python"; V="${1:?variant}"
case "$V" in B) CW=2000; HDR="";; C) CW=160; HDR=1;; D) CW=400; HDR="";; *) echo "variant must be B, C or D"; exit 2;; esac
export HF_HOME="$HOME/Library/Application Support/MediaFind/models/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DATA="data_s2_$V"; stage() { echo "=== $1 ($(date +%H:%M))"; }
rm -f "runs/s2_$V.DONE"
if [ ! -s "runs/s2_$V.lex.rank.json" ]; then stage "O1a ingest (chunk $CW words, header ${HDR:-0}) + rank original queries k=100"
  MF_DOC_CHUNK_WORDS=$CW MF_DOC_HEADER="$HDR" MEDIAFIND_EMBED_DEVICE=cpu ./run_rank.sh "$DATA" atm/atm-bench-san.doc.jsonl "runs/s2_$V" lex:lexical:k=100 > "runs/s2_$V.log" 2>&1; grep -E "RESULT|ingest\]|chunk\]" "runs/s2_$V.log"; fi
if [ ! -s "runs/s2_${V}_rw.lex.rank.json" ]; then stage "O1b rank rewritten queries k=100"
  MF_DOC_CHUNK_WORDS=$CW MF_DOC_HEADER="$HDR" MEDIAFIND_EMBED_DEVICE=cpu ./run_rank.sh "$DATA" atm/atm-bench-san-rw.doc.jsonl "runs/s2_${V}_rw" lex:lexical:k=100 > "runs/s2_${V}_rw.log" 2>&1; grep -E "RESULT|ingest\]" "runs/s2_${V}_rw.log"; fi
[ -s "runs/s2_$V.lex.rank.json" ] && [ -s "runs/s2_${V}_rw.lex.rank.json" ] || { echo "ABORT: a k=100 ranking is missing"; touch "runs/s2_$V.DONE"; exit 1; }
if [ ! -s "runs/s2_$V.ce50.rank.json" ]; then stage "O2 cross-encoder over the original-query top-50 (MPS, CPU fallback)"
  ITEMS_FILE="$SP/atm/items-san.json" CE_DEVICE=mps CE_BS=4 /Users/mbp/code/mediafind/.venv/bin/python -u ce_screen.py "runs/s2_$V.lex.rank.json" "runs/s2_$V.ce50.rank.json" BAAI/bge-reranker-v2-m3 50 1024 > "runs/s2_${V}_ce50.log" 2>&1; tail -1 "runs/s2_${V}_ce50.log"; fi
[ -s "runs/s2_$V.ce50.rank.json" ] || { echo "ABORT: CE ranking missing (see runs/s2_${V}_ce50.log)"; touch "runs/s2_$V.DONE"; exit 1; }
stage "fusion + gate"
V="$V" "$PY" -I - <<'PY' | tee "runs/s2_${V}_gate.txt"
import json, os
V = os.environ["V"]
qa = json.load(open("atm/atm-bench.json")); byid = {q["id"]: q for q in qa}
key = json.load(open("atm/rw-san-key.json")); RW0 = json.load(open(f"runs/s2_{V}_rw.lex.rank.json")); RW = {byid[key[k]]["question"]: v for k, v in RW0.items() if k in key}
CE = json.load(open(f"runs/s2_{V}.ce50.rank.json")); LEX = json.load(open(f"runs/s2_{V}.lex.rank.json")); OLD = json.load(open("runs/san_fused.rank.json"))
def rrf(*Rs, k=10, depth=100):
    out = {}
    for q in qa:
        sc = {}
        for R in Rs:
            for r, i in enumerate(R.get(q["question"], [])[:depth]): sc[i] = sc.get(i, 0) + 1 / (k + r + 1)
        out[q["question"]] = [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])]
    return out
F = rrf(RW, CE); json.dump(F, open(f"runs/san_fused_s2_{V}.rank.json", "w"))
def R(F, k): return 100 * sum(sum(e in F[q["question"]][:k] for e in q["evidence_ids"]) / len(q["evidence_ids"]) for q in qa) / len(qa)
cov = lambda F, q: set(q["evidence_ids"]) <= set(F[q["question"]][:25])
unc = sum(not cov(F, q) for q in qa); lost = sum(cov(OLD, q) and not cov(F, q) for q in qa); new = sum(not cov(OLD, q) and cov(F, q) for q in qa)
for name, X in (("lex100 (new)", LEX), ("rw100 (new)", RW), ("ce50 (new)", CE)):
    print(f"{name:14s} " + "  ".join("R@%d %.2f" % (k, R(X, k)) for k in (1, 5, 10, 25, 50)))
print(f"S2-{V} fused: " + "  ".join("R@%d %.2f" % (k, R(F, k)) for k in (1, 5, 10, 25, 50)) + f" | uncovered@25 {unc} (was 86): newly covered {new}, lost coverage {lost}")
print("submitted:   " + "  ".join("R@%d %.2f" % (k, R(OLD, k)) for k in (1, 5, 10, 25, 50)))
ok = unc <= 71 and R(F, 10) >= 89.43
print("GATE", "PASS" if ok else "FAIL", "(rule: uncovered <= 71, R@10 >= 89.43)")
PY
touch "runs/s2_$V.DONE"; stage "S2-$V local stages done"

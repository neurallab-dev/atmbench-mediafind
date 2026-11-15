#!/usr/bin/env bash
# R5 local stages: deeper first stage (k=200, original + rewritten queries) -> union candidate pool (RRF k=10) -> CE over the
# top-150 of the pool -> fusion RRF k=10 depth 100 (rw200, ce150), same structure as the submitted ranking -> deterministic gate.
# Pre-stated gate: uncovered@25 <= 71 (now 86) AND R@10 >= 89.43 (now 89.73, -0.3) AND previously-covered q losing coverage <= 5.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"; PY="$HOME/atmbench/.venv/bin/python"
export HF_HOME="$HOME/Library/Application Support/MediaFind/models/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
stage() { echo "=== $1 ($(date +%H:%M))"; }
rm -f runs/r5_local.DONE
if [ ! -s runs/san200.lex200.rank.json ]; then stage "O1a first stage k=200, original queries"
  MEDIAFIND_EMBED_DEVICE=cpu ./run_rank.sh data_san atm/atm-bench-san.doc.jsonl runs/san200 lex200:lexical:k=200 > runs/san200.log 2>&1; grep -E "RESULT|ingest\]" runs/san200.log; fi
if [ ! -s runs/rw_san200.lex200.rank.json ]; then stage "O1b first stage k=200, rewritten queries"
  MEDIAFIND_EMBED_DEVICE=cpu ./run_rank.sh data_san atm/atm-bench-san-rw.doc.jsonl runs/rw_san200 lex200:lexical:k=200 > runs/rw_san200.log 2>&1; grep -E "RESULT|ingest\]" runs/rw_san200.log; fi
[ -s runs/san200.lex200.rank.json ] && [ -s runs/rw_san200.lex200.rank.json ] || { echo "ABORT: a k=200 ranking is missing"; touch runs/r5_local.DONE; exit 1; }
stage "union candidate pool"
"$PY" -I - <<'PY'
import json, collections
qa = json.load(open("atm/atm-bench.json")); byid = {q["id"]: q for q in qa}
key = json.load(open("atm/rw-san-key.json")); RW0 = json.load(open("runs/rw_san200.lex200.rank.json")); LEX = json.load(open("runs/san200.lex200.rank.json"))
RW = {byid[key[k]]["question"]: v for k, v in RW0.items() if k in key}
items = {i["item_id"] for i in json.load(open("atm/items-san.json"))}
print("depths: lex200 %s rw200 %s; rw mapped %d/%d" % (collections.Counter(len(v) for v in LEX.values()).most_common(1), collections.Counter(len(v) for v in RW.values()).most_common(1), len(RW), len(qa)))
pool = {}
for q in qa:
    sc = {}
    for R in (LEX, RW):
        for r, i in enumerate(R.get(q["question"], [])):
            if i in items: sc[i] = sc.get(i, 0) + 1 / (10 + r + 1)
    pool[q["question"]] = [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])]
json.dump(pool, open("runs/r5_pool.rank.json", "w"))
reach = lambda n: sum(set(q["evidence_ids"]) <= set(pool[q["question"]][:n]) for q in qa)
print("pool sizes mean %.0f max %d; questions fully reachable within pool top-150: %d, top-200: %d, whole pool: %d (of 1013; covered@25 now 927)" % (sum(len(v) for v in pool.values()) / len(pool), max(len(v) for v in pool.values()), reach(150), reach(200), reach(10**6)))
PY
if [ ! -s runs/r5_ce150.rank.json ]; then stage "O2 cross-encoder over pool top-150 (MPS)"
  ITEMS_FILE="$SP/atm/items-san.json" CE_DEVICE=mps CE_BS=4 /Users/mbp/code/mediafind/.venv/bin/python -u ce_screen.py runs/r5_pool.rank.json runs/r5_ce150.rank.json BAAI/bge-reranker-v2-m3 150 1024 > runs/r5_ce150.log 2>&1; tail -1 runs/r5_ce150.log; fi
[ -s runs/r5_ce150.rank.json ] || { echo "ABORT: CE ranking missing (see runs/r5_ce150.log)"; touch runs/r5_local.DONE; exit 1; }
stage "fusion + gate"
"$PY" -I - <<'PY' | tee runs/r5_gate.txt
import json
qa = json.load(open("atm/atm-bench.json")); byid = {q["id"]: q for q in qa}
key = json.load(open("atm/rw-san-key.json")); RW0 = json.load(open("runs/rw_san200.lex200.rank.json")); RW = {byid[key[k]]["question"]: v for k, v in RW0.items() if k in key}
CE = json.load(open("runs/r5_ce150.rank.json")); OLD = json.load(open("runs/san_fused.rank.json"))
def rrf(*Rs, k=10, depth=100):
    out = {}
    for q in qa:
        sc = {}
        for R in Rs:
            for r, i in enumerate(R.get(q["question"], [])[:depth]): sc[i] = sc.get(i, 0) + 1 / (k + r + 1)
        out[q["question"]] = [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])]
    return out
F = rrf(RW, CE); json.dump(F, open("runs/san_fused_r5.rank.json", "w"))
def R(F, k): return 100 * sum(sum(e in F[q["question"]][:k] for e in q["evidence_ids"]) / len(q["evidence_ids"]) for q in qa) / len(qa)
cov = lambda F, q: set(q["evidence_ids"]) <= set(F[q["question"]][:25])
unc = sum(not cov(F, q) for q in qa); lost = sum(cov(OLD, q) and not cov(F, q) for q in qa); new = sum(not cov(OLD, q) and cov(F, q) for q in qa)
changed = sum(set(OLD[q["question"]][:25]) != set(F[q["question"]][:25]) for q in qa)
print("R5 fused: " + "  ".join("R@%d %.2f" % (k, R(F, k)) for k in (1, 5, 10, 25, 50)) + " | uncovered@25 %d (was 86): newly covered %d, lost coverage %d; top-25 set changed for %d q" % (unc, new, lost, changed))
print("submitted: " + "  ".join("R@%d %.2f" % (k, R(OLD, k)) for k in (1, 5, 10, 25, 50)))
ok = unc <= 71 and R(F, 10) >= 89.43 and lost <= 5
print("GATE", "PASS" if ok else "FAIL", "(rule: uncovered <= 71, R@10 >= 89.43, lost <= 5)")
PY
touch runs/r5_local.DONE
stage "R5 local stages done"

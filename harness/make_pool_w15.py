"""List and number pool (verbatim from the inline fusion of 2026-10-29 08:19 UTC): RRF k=10, depth 100, over the
Qwen3-Embedding-4B lexical rankings of the three rewrite sets, the bge-m3 rerank list (weight 1.5) and the 4B rerank
list. -> runs/san_fused_e_q4b_w15.rank.json"""
import json, collections, os
SP = os.path.dirname(os.path.abspath(__file__)); os.chdir(SP)
TAG = "q4b"; RWT = ["san", "sanmini", "sangpt54"]
qa = json.load(open("atm/atm-bench.json")); byid = {q["id"]: q for q in qa}
def load_rw(p, keyp):
    key = json.load(open(keyp)); R = json.load(open(p)); return {byid[key[k]]["question"]: v for k, v in R.items() if k in key}
OLD = json.load(open("runs/san_fused.rank.json")); OLDCE = json.load(open("runs/san_full.ce50.rank.json")); CE = json.load(open(f"runs/e_{TAG}.ce50.rank.json"))
RW = [load_rw(f"runs/e_{TAG}_rw_{t}.lex.rank.json", f"atm/rw-{t}-key.json") for t in RWT]
Rs = RW + [OLDCE, CE]; w = [1, 1, 1, 1.5, 1]; out = {}
for q in qa:
    sc = collections.defaultdict(float)
    for j, R in enumerate(Rs):
        for r, i in enumerate(R.get(q["question"], [])[:100]): sc[i] += w[j] / (10 + r + 1)
    out[q["question"]] = [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])]
json.dump(out, open("runs/san_fused_e_q4b_w15.rank.json", "w"))
def R_(F, k): return 100 * sum(sum(e in F.get(q["question"], [])[:k] for e in q["evidence_ids"]) / len(q["evidence_ids"]) for q in qa) / len(qa)
cov = lambda F, q: set(q["evidence_ids"]) <= set(F.get(q["question"], [])[:25])
print("w15: R@10 %.2f R@25 %.2f unc@25 %d lost %d" % (R_(out, 10), R_(out, 25), sum(not cov(out, q) for q in qa), sum(cov(OLD, q) and not cov(out, q) for q in qa)))

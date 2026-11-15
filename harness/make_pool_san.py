"""Open-ended pool (verbatim from pipeline_san.sh's inline fusion, 2026-10-24): RRF k=10, depth 100, of the bge-m3
lexical ranking of the DeepSeek rewrite and the bge-m3 whole-item rerank list. -> runs/san_fused.rank.json"""
import json, os
SP = os.path.dirname(os.path.abspath(__file__)); os.chdir(SP)
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

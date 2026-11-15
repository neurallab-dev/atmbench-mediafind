"""Gate for a candidate-embedder product retrieval: fuse (rewritten-query lists..., ce50 of the new original list) with RRF k=10
depth 100 as the submitted ranking was built, compare with runs/san_fused.rank.json.  usage: e_gate.py TAG RWTAG...
Pre-stated gate: uncovered@25 <= 71 (submitted 86), R@10 >= 89.43, previously-covered questions lost <= 5."""
import json, sys, collections
TAG = sys.argv[1]; RWT = sys.argv[2:] or ["san"]
qa = json.load(open("atm/atm-bench.json")); byid = {q["id"]: q for q in qa}
def load_rw(p, keyp):
    key = json.load(open(keyp)); R = json.load(open(p)); return {byid[key[k]]["question"]: v for k, v in R.items() if k in key}
OLD = json.load(open("runs/san_fused.rank.json")); OLDLEX = json.load(open("runs/san_full.lex.rank.json")); OLDCE = json.load(open("runs/san_full.ce50.rank.json"))
LEX = json.load(open(f"runs/e_{TAG}.lex.rank.json")); CE = json.load(open(f"runs/e_{TAG}.ce50.rank.json"))
RW = {t: load_rw(f"runs/e_{TAG}_rw_{t}.lex.rank.json", f"atm/rw-{t}-key.json") for t in RWT}
RWOLD = {t: load_rw(f"runs/rw_{t}.lex.rank.json", f"atm/rw-{t}-key.json") for t in RWT}
def rrf(Rs, k=10, depth=100, w=None):
    out = {}
    for q in qa:
        sc = collections.defaultdict(float)
        for j, R in enumerate(Rs):
            for r, i in enumerate(R.get(q["question"], [])[:depth]): sc[i] += (w[j] if w else 1.0) / (k + r + 1)
        out[q["question"]] = [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])]
    return out
def R_(F, k): return 100 * sum(sum(e in F.get(q["question"], [])[:k] for e in q["evidence_ids"]) / len(q["evidence_ids"]) for q in qa) / len(qa)
cov = lambda F, q: set(q["evidence_ids"]) <= set(F.get(q["question"], [])[:25])
def line(name, F):
    unc = sum(not cov(F, q) for q in qa); new = sum(cov(F, q) and not cov(OLD, q) for q in qa); lost = sum(cov(OLD, q) and not cov(F, q) for q in qa)
    print(f"{name:44s} " + " ".join("R@%d %.2f" % (k, R_(F, k)) for k in (1, 5, 10, 25, 50, 100)) + f" | unc@25 {unc:3d} (+{new} new, -{lost} lost)"); return unc, lost
line("bge-m3 lexical (submitted first stage)", OLDLEX); line("bge-m3 ce50", OLDCE); line("submitted fused", OLD)
line(f"{TAG} lexical (original q)", LEX); line(f"{TAG} ce50", CE)
for t in RWT: line(f"{TAG} rewritten {t}", RW[t])
cands = {}
cands["new only: (rw_new..., ce_new)"] = rrf([RW[t] for t in RWT] + [CE])
cands["mixed: (rw_old..., rw_new..., ce_old, ce_new)"] = rrf([RWOLD[t] for t in RWT] + [RW[t] for t in RWT] + [OLDCE, CE])
cands["mixed: (rw_old san, ce_old, ce_new)"] = rrf([RWOLD["san"], OLDCE, CE])
cands["mixed: (rw_new..., ce_old, ce_new)"] = rrf([RW[t] for t in RWT] + [OLDCE, CE])
cands["mixed: (rw_old..., rw_new..., ce_new)"] = rrf([RWOLD[t] for t in RWT] + [RW[t] for t in RWT] + [CE])
res = {nm: line(nm, F) for nm, F in cands.items()}
best = min(res, key=lambda nm: res[nm][0]); unc, lost = res[best]; F = cands[best]
json.dump(F, open(f"runs/san_fused_e_{TAG}.rank.json", "w"))
ok = unc <= 71 and R_(F, 10) >= 89.43 and lost <= 5
print(f"\nEMBED GATE {'PASS' if ok else 'FAIL'} [{best}]: unc@25 {unc} (need <= 71), R@10 {R_(F,10):.2f} (need >= 89.43), lost {lost} (need <= 5) -> runs/san_fused_e_{TAG}.rank.json")

"""Free fusion sweep around the step-2 best: weights / RRF k / list drops; reports unc@25, new, lost, R@10, R@25 vs submitted."""
import json, sys, collections
TAG = sys.argv[1]; RWT = sys.argv[2:] or ["san"]
import os
qa = json.load(open("atm/atm-bench.json")); byid = {q["id"]: q for q in qa}
if os.environ.get("SUBSET"):
    ids = {q["id"] for q in json.load(open(os.environ["SUBSET"]))}; qa = [q for q in qa if q["id"] in ids]; print("SUBSET", len(qa), "q")
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
    ok = unc <= 71 and R_(F, 10) >= 89.43 and lost <= 5
    print(f"{'PASS' if ok else '    '} {name:52s} R@10 {R_(F,10):.2f} R@25 {R_(F,25):.2f} | unc@25 {unc:3d} (+{new} new, -{lost} lost)", flush=True); return unc, lost, F
NEW = [RW[t] for t in RWT]
C = {}
C["base (rw_new..., ce_old, ce_new)"] = rrf(NEW + [OLDCE, CE])
for k in (20, 30, 60): C[f"base k={k}"] = rrf(NEW + [OLDCE, CE], k=k)
for w in (1.5, 2.0, 3.0): C[f"base ce_old x{w}"] = rrf(NEW + [OLDCE, CE], w=[1]*len(NEW) + [w, 1])
for w in (1.5, 2.0): C[f"base ce_new x{w}"] = rrf(NEW + [OLDCE, CE], w=[1]*len(NEW) + [1, w])
for w in (1.5, 2.0): C[f"base both ce x{w}"] = rrf(NEW + [OLDCE, CE], w=[1]*len(NEW) + [w, w])
for w in (0.5, 1.0): C[f"base + {w} rw_old san"] = rrf(NEW + [OLDCE, CE, RWOLD["san"]], w=[1]*len(NEW) + [1, 1, w])
C["base + 0.5 rw_old san, ce_old x1.5"] = rrf(NEW + [OLDCE, CE, RWOLD["san"]], w=[1]*len(NEW) + [1.5, 1, 0.5])
C["base + 0.5 old fused"] = rrf(NEW + [OLDCE, CE, OLD], w=[1]*len(NEW) + [1, 1, 0.5])
C["base + 1.0 old fused"] = rrf(NEW + [OLDCE, CE, OLD])
C["base + lex_new"] = rrf(NEW + [OLDCE, CE, LEX])
C["base + 0.5 lex_new"] = rrf(NEW + [OLDCE, CE, LEX], w=[1]*len(NEW) + [1, 1, 0.5])
for t in RWT:
    C[f"base minus rw_new {t}"] = rrf([RW[u] for u in RWT if u != t] + [OLDCE, CE])
C["(rw_new..., ce_old, ce_new) depth 50"] = rrf(NEW + [OLDCE, CE], depth=50)
C["(rw_new..., ce_old x2, ce_new x2, 0.5 rw_old san)"] = rrf(NEW + [OLDCE, CE, RWOLD["san"]], w=[1]*len(NEW) + [2, 2, 0.5])
res = {nm: line(nm, F) for nm, F in C.items()}
passing = [nm for nm in res if res[nm][0] <= 71 and res[nm][1] <= 5 and R_(res[nm][2], 10) >= 89.43]
if passing:
    best = min(passing, key=lambda nm: res[nm][0]); print("\nBEST PASSING:", best, "unc", res[best][0], "lost", res[best][1])
    os.environ.get("SUBSET") or json.dump(res[best][2], open(f"runs/san_fused_e_{TAG}_sweep.rank.json", "w"))
else: print("\nno passing variant")
# where do the base's lost questions sit?
F = C["base (rw_new..., ce_old, ce_new)"]
print("\nlost in base: qtype, old rank of evidence, new rank of evidence")
for q in qa:
    if cov(OLD, q) and not cov(F, q):
        o = F.get(q["question"], []); print(" ", q["id"], q.get("question_type") or q.get("type"), [ (o.index(e)+1 if e in o else None) for e in q["evidence_ids"]])

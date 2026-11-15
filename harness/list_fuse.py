"""List-only fusion sweep adding the Qwen3-Reranker-4B list (runs/qr4b_list.rank.json) to the adopted list ranking.
PRE-REGISTERED GATE (139 list q, vs adopted runs/san_fused_e_q4b_w15.rank.json): unc@25 <= 8 (adopted 10, i.e. >= 2 net newly covered),
lost <= 1, R@10 >= 89.7 (adopted 90.21 - 0.5). Only a passing variant goes to a paid list-only answer pass."""
import json, sys, collections, os
qa = json.load(open("atm/atm-bench-list-only.json")); byid = {q["id"]: q for q in json.load(open("atm/atm-bench.json"))}
assert len(qa) == 139
def load_rw(p, keyp):
    key = json.load(open(keyp)); R = json.load(open(p)); return {byid[key[k]]["question"]: v for k, v in R.items() if k in key}
ADOPT = json.load(open("runs/san_fused_e_q4b_w15.rank.json"))
OLDCE = json.load(open("runs/san_full.ce50.rank.json")); CE = json.load(open("runs/e_q4b.ce50.rank.json"))
QR = json.load(open("runs/qr4b_list.rank.json")); LEX = json.load(open("runs/e_q4b.lex.rank.json"))
RWT = ["san", "sanmini", "sangpt54"]
RW = {t: load_rw(f"runs/e_q4b_rw_{t}.lex.rank.json", f"atm/rw-{t}-key.json") for t in RWT}
NEW = [RW[t] for t in RWT]
missing = [q["id"] for q in qa if q["question"] not in QR]; assert not missing, missing
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
    unc = sum(not cov(F, q) for q in qa); new = sum(cov(F, q) and not cov(ADOPT, q) for q in qa); lost = sum(cov(ADOPT, q) and not cov(F, q) for q in qa)
    ok = unc <= 8 and lost <= 1 and R_(F, 10) >= 89.7
    print(f"{'PASS' if ok else '    '} {name:50s} R@10 {R_(F,10):.2f} R@25 {R_(F,25):.2f} | unc@25 {unc:3d} (+{new} new, -{lost} lost)", flush=True); return unc, lost, F
line("ADOPTED (file)", ADOPT); line("qr alone", QR); line("ce_new alone", CE); line("ce_old alone", OLDCE)
C = {}
C["adopted recomputed (rw3, ce_old x1.5, ce_new)"] = rrf(NEW + [OLDCE, CE], w=[1, 1, 1, 1.5, 1])
for w in (1.0, 1.5, 2.0, 3.0): C[f"adopted + qr x{w}"] = rrf(NEW + [OLDCE, CE, QR], w=[1, 1, 1, 1.5, 1, w])
for w in (1.0, 1.5, 2.0): C[f"rw3, ce_old x1.5, qr x{w} (no ce_new)"] = rrf(NEW + [OLDCE, QR], w=[1, 1, 1, 1.5, w])
for w in (1.5, 2.0, 3.0): C[f"rw3, qr x{w} (no ce)"] = rrf(NEW + [QR], w=[1, 1, 1, w])
C["rw3, ce_old x1.5, ce_new x1.5, qr x1.5"] = rrf(NEW + [OLDCE, CE, QR], w=[1, 1, 1, 1.5, 1.5, 1.5])
C["rw3, ce_old, ce_new, qr (all x1)"] = rrf(NEW + [OLDCE, CE, QR])
for k in (20, 30): C[f"adopted + qr x1.5 k={k}"] = rrf(NEW + [OLDCE, CE, QR], k=k, w=[1, 1, 1, 1.5, 1, 1.5])
C["adopted + qr x1.5 + lex_new x0.5"] = rrf(NEW + [OLDCE, CE, QR, LEX], w=[1, 1, 1, 1.5, 1, 1.5, 0.5])
C["adopted + qr x2 depth 50"] = rrf(NEW + [OLDCE, CE, QR], depth=50, w=[1, 1, 1, 1.5, 1, 2])
res = {nm: line(nm, F) for nm, F in C.items()}
passing = [nm for nm in res if res[nm][0] <= 8 and res[nm][1] <= 1 and R_(res[nm][2], 10) >= 89.7]
if passing:
    best = min(passing, key=lambda nm: (res[nm][0], -R_(res[nm][2], 10))); print("\nBEST PASSING:", best, "unc", res[best][0], "lost", res[best][1])
    F = res[best][2]
    # full-set rank file: adopted ranking everywhere, the new fusion on the 139 list q
    out = dict(ADOPT); out.update(F); json.dump(out, open("runs/san_fused_list_qr.rank.json", "w")); print("wrote runs/san_fused_list_qr.rank.json")
    print("\nnewly covered / lost vs adopted (id, n_gold, adopted ranks, new ranks):")
    for q in qa:
        if cov(F, q) != cov(ADOPT, q):
            a = ADOPT.get(q["question"], []); n = F.get(q["question"], [])
            print(" ", "NEW " if cov(F, q) else "LOST", q["id"], len(q["evidence_ids"]), [(a.index(e)+1 if e in a else None) for e in q["evidence_ids"]], [(n.index(e)+1 if e in n else None) for e in q["evidence_ids"]])
else: print("\nno passing variant")
print("\nstill uncovered in adopted (id, n_gold, adopted ranks, qr-alone ranks):")
for q in qa:
    if not cov(ADOPT, q):
        a = ADOPT.get(q["question"], []); n = QR.get(q["question"], [])
        print(" ", q["id"], len(q["evidence_ids"]), [(a.index(e)+1 if e in a else None) for e in q["evidence_ids"]], [(n.index(e)+1 if e in n else None) for e in q["evidence_ids"]])

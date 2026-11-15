"""Gate for a new rewritten-query list: fuse it with the submitted CE50 list exactly as the submitted ranking was built
(RRF k=10 depth 100 of (rw, ce50)) and compare with runs/san_fused.rank.json.
usage: rw_gate.py TAG   (reads runs/rw_TAG.lex.rank.json + atm/rw-TAG-key.json; writes runs/san_fused_TAG.rank.json and _3.rank.json)
Pre-stated gate: uncovered@25 <= 76 (submitted 86), R@10 >= 89.43, previously-covered questions lost <= 5."""
import json, sys, collections
TAG = sys.argv[1]
qa = json.load(open("atm/atm-bench.json")); byid = {q["id"]: q for q in qa}
def load_rw(tag):
    key = json.load(open(f"atm/rw-{tag}-key.json")); R = json.load(open(f"runs/rw_{tag}.lex.rank.json"))
    return {byid[key[k]]["question"]: v for k, v in R.items() if k in key}
RW_OLD = load_rw("san"); RW = load_rw(TAG); CE = json.load(open("runs/san_full.ce50.rank.json")); OLD = json.load(open("runs/san_fused.rank.json"))
def rrf(*Rs, k=10, depth=100, w=None):
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
    return f"{name:34s} " + " ".join("R@%d %.2f" % (k, R_(F, k)) for k in (1, 5, 10, 25, 50)) + f" | unc@25 {unc:3d} (+{new} new, -{lost} lost)"
print(line("rw san (submitted rewrites) alone", RW_OLD)); print(line(f"rw {TAG} alone", RW)); print(line("submitted fused (rw san, ce50)", OLD))
F2 = rrf(RW, CE); F3 = rrf(RW_OLD, RW, CE); F3h = rrf(RW_OLD, RW, CE, w=[0.5, 0.5, 1.0])
for nm, F in ((f"fused (rw {TAG}, ce50)", F2), (f"fused (rw san, rw {TAG}, ce50)", F3), (f"fused (0.5 rw san, 0.5 rw {TAG}, ce50)", F3h)): print(line(nm, F))
json.dump(F2, open(f"runs/san_fused_{TAG}.rank.json", "w")); json.dump(F3, open(f"runs/san_fused_{TAG}_3.rank.json", "w"))
best = max(((nm, F) for nm, F in (("2", F2), ("3", F3), ("3h", F3h))), key=lambda x: -sum(not cov(x[1], q) for q in qa))
nm, F = best; unc = sum(not cov(F, q) for q in qa); lost = sum(cov(OLD, q) and not cov(F, q) for q in qa)
ok = unc <= 76 and R_(F, 10) >= 89.43 and lost <= 5
print(f"\nRW GATE {'PASS' if ok else 'FAIL'} (variant {nm}): unc@25 {unc} (need <= 76), R@10 {R_(F,10):.2f} (need >= 89.43), lost {lost} (need <= 5)")

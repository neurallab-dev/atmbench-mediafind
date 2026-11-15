"""Write the full-set rank file for the pre-registered list variant: adopted ranking everywhere, and on the 139 list q
RRF k=10 depth 100 of [rw_new san, sanmini, sangpt54, ce_old x1.5, ce_new, qr4b x3]."""
import json, collections
qa = json.load(open("atm/atm-bench-list-only.json")); byid = {q["id"]: q for q in json.load(open("atm/atm-bench.json"))}
def load_rw(p, keyp):
    key = json.load(open(keyp)); R = json.load(open(p)); return {byid[key[k]]["question"]: v for k, v in R.items() if k in key}
ADOPT = json.load(open("runs/san_fused_e_q4b_w15.rank.json"))
OLDCE = json.load(open("runs/san_full.ce50.rank.json")); CE = json.load(open("runs/e_q4b.ce50.rank.json")); QR = json.load(open("runs/qr4b_list.rank.json"))
RW = [load_rw(f"runs/e_q4b_rw_{t}.lex.rank.json", f"atm/rw-{t}-key.json") for t in ("san", "sanmini", "sangpt54")]
Rs = RW + [OLDCE, CE, QR]; w = [1, 1, 1, 1.5, 1, 3]
out = dict(ADOPT); n = 0
for q in qa:
    sc = collections.defaultdict(float)
    for j, R in enumerate(Rs):
        for r, i in enumerate(R.get(q["question"], [])[:100]): sc[i] += w[j] / (10 + r + 1)
    out[q["question"]] = [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])]; n += 1
json.dump(out, open("runs/san_fused_list_qr3.rank.json", "w"))
R_ = lambda F, k: 100 * sum(sum(e in F.get(q["question"], [])[:k] for e in q["evidence_ids"]) / len(q["evidence_ids"]) for q in qa) / len(qa)
cov = lambda F, q: set(q["evidence_ids"]) <= set(F.get(q["question"], [])[:25])
print("replaced", n, "list q; list R@10 %.2f R@25 %.2f unc@25 %d | adopted R@10 %.2f R@25 %.2f unc@25 %d" % (R_(out, 10), R_(out, 25), sum(not cov(out, q) for q in qa), R_(ADOPT, 10), R_(ADOPT, 25), sum(not cov(ADOPT, q) for q in qa)))
print("non-list q unchanged:", all(out[k] == ADOPT[k] for k in ADOPT if k not in {q["question"] for q in qa}))

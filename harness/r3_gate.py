"""R3 subset gate: run ONLY the second opinion (no reduce, no judge) and score it against the evidence ids.

usage: SECOND_MODEL=.. [SECOND_K=5] [SECOND_CTX=1] r3_gate.py --qa atm/atm-subset-sha1.json --pool runs/san_fused.rank.json --topk 25 --workers 6
Prints: recall of restorable questions (r), share of non-gold rejected items kept (p), predicted full-set net
correct = 16.3*r - 22*p (from the 10-27 sizing: 65 restorable q x +0.25, 3,287 non-gold x -0.0067), and the cost line.
"""
import argparse, json, os, sys, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mf_answer_r3 as R
M = R.M
ap = argparse.ArgumentParser(); ap.add_argument("--qa", required=True); ap.add_argument("--pool", required=True)
ap.add_argument("--topk", type=int, default=25); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--out")
a = ap.parse_args()
qa = json.load(open(a.qa)); rank = json.load(open(a.pool))
def one(q):
    pool = [i for i in rank[q["question"]][:a.topk] if i in M.ITEMS]
    kept, rejected, second = R.decide(q, pool)
    return q, {i for i, _ in kept}, rejected, {i for i, _ in second}
rows = []
with cf.ThreadPoolExecutor(a.workers) as ex:
    for q, kept, rejected, second in ex.map(one, qa):
        rows.append((q, kept, rejected, second))
gold_rej = [(q, i) for q, kept, rej, sec in rows for i in rej if i in q["evidence_ids"]]
non_rej = [(q, i) for q, kept, rej, sec in rows for i in rej if i not in q["evidence_ids"]]
restorable = [(q, kept, rej, sec) for q, kept, rej, sec in rows
              if not set(q["evidence_ids"]) <= kept and set(q["evidence_ids"]) <= (kept | set(rej))]
r_q = sum(1 for q, kept, rej, sec in restorable if set(q["evidence_ids"]) <= (kept | sec)) / max(1, len(restorable))
r_i = sum(1 for q, i in gold_rej if i in next(s for qq, k, rj, s in rows if qq is q)) / max(1, len(gold_rej))
p = sum(1 for q, i in non_rej if i in next(s for qq, k, rj, s in rows if qq is q)) / max(1, len(non_rej))
changed = sum(1 for q, kept, rej, sec in rows if sec)
print(f"second opinion {R.SECOND_MODEL} K={R.K} ctx={R.SECOND_CTX} on {len(rows)} q: rejected items {sum(len(rj) for _,_,rj,_ in rows)}, "
      f"gold among them {len(gold_rej)} ({len(restorable)} restorable q)")
print(f"  restorable-q recall r = {r_q:.2f}  (gold-item recall {r_i:.2f});  non-gold kept p = {p:.3f} ({sum(1 for q,i in non_rej if i in next(s for qq,k,rj,s in rows if qq is q))} of {len(non_rej)});  questions whose reduce input changes: {changed} ({changed/len(rows):.0%})")
print(f"  predicted full-set net = 16.3*r - 22*p = {16.3*r_q - 22*p:+.1f} correct ({(16.3*r_q - 22*p)/10.13:+.2f} QS); with a 2x tax on selected distractors: {16.3*r_q - 44*p:+.1f} correct")
print("  " + M.llm.cost_line())
if a.out:
    json.dump([{"id": q["id"], "kept": sorted(kept), "rejected": rej, "second": sorted(sec)} for q, kept, rej, sec in rows], open(a.out, "w"), indent=0)

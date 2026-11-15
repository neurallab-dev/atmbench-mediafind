"""Subset map-only test: one 25-item map call per question vs the cached 10-item batches. Deterministic kept stats, no reduce.
usage: map_batch_test.py RANK TAG [BATCH]"""
import os, sys, json, collections, statistics as st, concurrent.futures as cf
os.environ["ITEMS_FILE"] = "atm/items-san.json"; os.environ["BEST_GUESS"] = "1"; os.environ.setdefault("MAP_MODEL", "deepseek/deepseek-v4.1-flash"); os.environ["ANSWER_MODEL"] = "openai/gpt-5.4-mini"
sys.path.insert(0, "."); import llm, mf_answer_v2 as M
RANK, TAG = sys.argv[1], sys.argv[2]; BIG = int(sys.argv[3]) if len(sys.argv) > 3 else 25
POOLK = int(os.environ.get("POOLK", "25")); BASE10 = os.environ.get("BASE10", "1") == "1"
rank = json.load(open(RANK)); sub = json.load(open(os.environ.get("QA_FILE", "atm/atm-subset-sha1.json")))
def run(q):
    pool = [i for i in rank.get(q["question"], [])[:POOLK] if i in M.ITEMS]
    k10 = []
    if BASE10:
        for s in range(0, len(pool), 10): k10 += [i for i, _ in M.map_batch(q["question"], pool[s:s+10])]
    kb = []
    for s in range(0, len(pool), BIG): kb += [i for i, _ in M.map_batch(q["question"], pool[s:s+BIG])]
    return q, pool, k10, kb
res = {}
with cf.ThreadPoolExecutor(6) as ex:
    for q, pool, k10, kb in ex.map(run, sub):
        res[q["id"]] = {"qtype": q["qtype"], "pool": pool, "k10": k10, "kb": kb, "gold": q["evidence_ids"]}
json.dump(res, open(f"runs/mapb_{TAG}_b{BIG}_k{POOLK}.json" if POOLK != 25 else f"runs/mapb_{TAG}_b{BIG}.json", "w"))
print(f"== {TAG} pool={RANK} depth {POOLK} batch 10 vs {BIG}  ({len(sub)} q)")
for t in ("open_end", "number", "list_recall", "ALL"):
    rows = [r for r in res.values() if t == "ALL" or r["qtype"] == t]
    if not rows: continue
    def stats(key):
        g = [len(set(r[key]) & set(r["gold"])) for r in rows]; d = [len(set(r[key]) - set(r["gold"])) for r in rows]
        allg = sum(len(set(r[key]) & set(r["gold"])) == len(set(r["gold"]) & set(r["pool"])) for r in rows)
        none = sum(len(r[key]) == 0 for r in rows)
        return f"kept {st.mean(g)+st.mean(d):.2f} gold {st.mean(g):.2f} distr {st.mean(d):.2f} all-gold-in-pool-kept {allg}/{len(rows)} kept-nothing {none}"
    print(f"{t:12s} n {len(rows):3d} | b10: {stats('k10')} | b{BIG}: {stats('kb')}")
print(llm.cost_line())

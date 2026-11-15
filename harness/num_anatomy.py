import json, re, collections
E = json.load(open("ans/san_full_q4blists.eval/atm_openai_gpt-5-mini.json"))
qa = {q["id"]: q for q in json.load(open("atm/atm-bench.json"))}
R = json.load(open("runs/san_fused_typecond.rank.json"))
num = [e for e in E if e["qtype"] == "number"]; wrong = [e for e in num if not e["accuracy"]]
print(len(wrong), "wrong of", len(num))
def nums(s): return [x.replace(",", "") for x in re.findall(r"-?\d[\d,]*\.?\d*", s or "")]
cats = collections.Counter(); rows = []
for e in wrong:
    q = qa[e["id"]]; pool = R.get(q["question"], [])[:25]
    inpool = all(g in pool for g in q["evidence_ids"])
    g = nums(e["ground_truth"]); p = nums(e["prediction"])
    if not inpool: cat = "A gold outside top-25"
    elif g and p and g[0] == p[0]: cat = "B same leading number, format differs"
    elif g and p and any(abs(float(x) - float(g[0])) <= 0.015 * max(1, abs(float(g[0]))) for x in p if x not in ("", "-", ".")): cat = "C within 1.5%"
    elif not p: cat = "D no number in prediction"
    else: cat = "E different number"
    cats[cat] += 1; rows.append((cat, e["ground_truth"], e["prediction"][:70], e.get("normalized_prediction", "")[:40], q["question"][:80]))
for r in sorted(rows): print(" | ".join(r))
print(); [print(f"{v:3d}  {k}") for k, v in sorted(cats.items())]

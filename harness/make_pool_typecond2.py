"""The ranking each answer used, per question type (as the board's evidence file reports it, 2026-10-29 15:36 UTC):
number and list_recall from the 4B fusion, open_end from the bge-m3 fusion. -> runs/san_fused_typecond2.rank.json"""
import json, os
SP = os.path.dirname(os.path.abspath(__file__)); os.chdir(SP)
qa = json.load(open("atm/atm-bench.json"))
OLD = json.load(open("runs/san_fused.rank.json")); W15 = json.load(open("runs/san_fused_e_q4b_w15.rank.json"))
out = {q["question"]: (W15[q["question"]] if q["qtype"] in ("number", "list_recall") else OLD[q["question"]]) for q in qa}
json.dump(out, open("runs/san_fused_typecond2.rank.json", "w"))
R_ = lambda F, k: 100 * sum(sum(e in F.get(q["question"], [])[:k] for e in q["evidence_ids"]) / len(q["evidence_ids"]) for q in qa) / len(qa)
print("typecond2: " + " ".join("R@%d %.2f" % (k, R_(out, k)) for k in (1, 5, 10, 25)))

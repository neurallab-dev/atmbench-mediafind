import sys, json
sys.path.insert(0, "/Users/mbp/atmbench/atm-src"); sys.path.insert(0, ".")
from memqa.utils.evaluator.evaluate_qa import deterministic_accuracy
from ans_format import format_number_answer as F
E = json.load(open("ans/san_full_q4blists.eval/atm_openai_gpt-5-mini.json")); qa = {q["id"]: q for q in json.load(open("atm/atm-bench.json"))}
num = [e for e in E if e["qtype"] == "number"]
flips, breaks, changed = [], [], 0
for e in num:
    p2 = F(e["prediction"])
    if p2 == e["prediction"]: continue
    changed += 1; a1 = bool(e["accuracy"]); a2 = deterministic_accuracy(e["ground_truth"], p2, qa[e["id"]]["question"])
    if a2 and not a1: flips.append((e["ground_truth"], e["prediction"], p2))
    if a1 and not a2: breaks.append((e["ground_truth"], e["prediction"], p2))
print(f"number q {len(num)} | changed {changed} | wrong->right {len(flips)} | right->wrong {len(breaks)}")
for f in flips: print("  FLIP ", f)
for b in breaks: print("  BREAK", b)
old = sum(bool(e["accuracy"]) for e in num); new = old + len(flips) - len(breaks)
print(f"number acc {100*old/len(num):.2f} -> {100*new/len(num):.2f} | QS delta on 1013 q: {100*(new-old)/1013:+.2f} (bar: > +0.30 over the board)")
print("\n== remaining wrong number q whose prediction still lacks a gold 'code' token (4+ chars) that the official scorer demands:")
import re
RE_CODE = re.compile(r"\b[A-Z0-9]{4,}\b", re.IGNORECASE)
n = 0
for e in num:
    p2 = F(e["prediction"])
    if deterministic_accuracy(e["ground_truth"], p2, qa[e["id"]]["question"]): continue
    gt_clean = re.sub(r"\([^)]+\)", "", e["ground_truth"]).rstrip(".")
    codes = [c.upper() for c in RE_CODE.findall(gt_clean)]
    missing = [c for c in codes if c not in p2.upper()]
    if missing: n += 1; print(f"  missing {missing} | {e['ground_truth']} | {p2[:60]}")
print(n, "q")

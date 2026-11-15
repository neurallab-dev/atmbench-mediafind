"""Probe the official deterministic scorer: recompute number accuracy, then test a 24-hour-clock post-normalisation of the predictions."""
import os, sys, json, re
sys.path.insert(0, os.environ.get("ATM_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "atm-official")))
from memqa.utils.evaluator.evaluate_qa import deterministic_accuracy
E = json.load(open("ans/san_full_q4blists.eval/atm_openai_gpt-5-mini.json"))
qa = {q["id"]: q for q in json.load(open("atm/atm-bench.json"))}
num = [e for e in E if e["qtype"] == "number"]
mism = [e["id"] for e in num if deterministic_accuracy(e["ground_truth"], e["prediction"], qa[e["id"]]["question"]) != bool(e["accuracy"])]
print("recompute mismatches vs stored:", len(mism), "of", len(num))
def to24(s):
    def rep(m):
        h = int(m.group(1)); mi = m.group(2) or "00"; ap = m.group(3).lower().replace(".", "")
        if h > 12: return m.group(0)
        if h == 12: h = 0
        if ap.startswith("p"): h += 12
        return f"{h:02d}:{mi}"
    return re.sub(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*([AaPp]\.?[Mm]\.?)(?![A-Za-z])", rep, s)
flips, breaks = [], []
for e in num:
    q = qa[e["id"]]["question"]; p2 = to24(e["prediction"])
    if p2 == e["prediction"]: continue
    a1 = bool(e["accuracy"]); a2 = deterministic_accuracy(e["ground_truth"], p2, q)
    if a2 and not a1: flips.append((e["ground_truth"], e["prediction"], p2))
    if a1 and not a2: breaks.append((e["ground_truth"], e["prediction"], p2))
print("changed predictions:", sum(to24(e["prediction"]) != e["prediction"] for e in num), "| wrong->right:", len(flips), "| right->wrong:", len(breaks))
for f in flips: print("  FLIP", f)
for b in breaks: print("  BREAK", b)
print("\n== puzzling pairs (why wrong?)")
for gt, pr, qid in [("April 20, 2023, at 14:40.", "2023-04-20 14:40", None), ("14:35 on February 22, 2025.", "2025-02-22 at 14:35", None), ("18:00", "6:00 PM", None), ("18:00", "18:00", None), ("March 28, 3PM", "2024-03-28 3:00 PM", None), ("March 28, 3PM", "2024-03-28 15:00", None), ("January 24, 2023, at 19:02.", "19:02", None), ("January 24, 2023, at 19:02.", "2023-01-24 19:02", None), ("500 pounds.", "£500", None), ("Around 25.", "about 25", None), ("February 1st from 3pm to 5:30pm", "Tuesday, February 1st, from 3 pm to 5:30 pm", None), ("February 1st from 3pm to 5:30pm", "February 1, 15:00 to 17:30", None)]:
    print(f"  {deterministic_accuracy(gt, pr, None)!s:5s} | {gt} | {pr}")
# the two real cases with their questions
for e in num:
    if e["ground_truth"] in ("April 20, 2023, at 14:40.", "14:35 on February 22, 2025."):
        print("  with question:", deterministic_accuracy(e["ground_truth"], e["prediction"], qa[e["id"]]["question"]), "|", qa[e["id"]]["question"][:90], "|", e["prediction"])

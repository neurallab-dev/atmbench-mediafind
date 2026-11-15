"""Apply the deterministic number-answer formatter (ans_format.py) to a judged run's number-type answers and re-score those with
the official deterministic scorer; list answers and the open_end judgments are carried over unchanged (no re-judging).
usage: fmt_eval.py NAME BASE_RUN"""
import json, os, sys, shutil
sys.path.insert(0, "/Users/mbp/atmbench/atm-src"); sys.path.insert(0, ".")
from memqa.utils.evaluator.evaluate_qa import _deterministic_accuracy_core
from ans_format import format_number_answer as F
NAME, BASE = sys.argv[1:3]
qa = json.load(open("atm/atm-bench.json")); Q = {q["id"]: q for q in qa}
A = {json.loads(l)["id"]: json.loads(l) for l in open(f"ans/{BASE}.jsonl")}
changed = 0
with open(f"ans/{NAME}.jsonl", "w") as f:
    for q in qa:
        r = dict(A[q["id"]])
        if q["qtype"] == "number":
            a2 = F(r["answer"]); changed += a2 != r["answer"]; r["answer"] = a2
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
ev = f"ans/{NAME}.eval"; shutil.rmtree(ev, ignore_errors=True); os.makedirs(ev)
flips = 0
for fn in ("atm_openai_gpt-5-mini.json", "deterministic_accuracy.json"):
    out = []
    for r in json.load(open(f"ans/{BASE}.eval/{fn}")):
        r = dict(r)
        if r["qtype"] == "number":
            p2 = F(r["prediction"]); ok, norm = _deterministic_accuracy_core(r["ground_truth"], p2, Q[r["id"]]["question"])
            if fn.startswith("atm"): flips += int(ok) - int(bool(r["accuracy"]))
            r["prediction"], r["normalized_prediction"], r["accuracy"] = p2, norm, bool(ok)
        out.append(r)
    json.dump(out, open(f"{ev}/{fn}", "w"), ensure_ascii=False, indent=1)
shutil.copy(f"ans/{BASE}.eval/llm_judge_openai_gpt-5-mini.json", f"{ev}/llm_judge_openai_gpt-5-mini.json")
J = json.load(open(f"{ev}/atm_openai_gpt-5-mini.json")); by = {}
for r in J: by.setdefault(r["qtype"], []).append(float(r["accuracy"]))
summary = {"count": len(J), "correct": sum(sum(v) for v in by.values()), "accuracy": sum(sum(v) for v in by.values()) / len(J),
           "by_qtype": {t: {"count": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)} for t, v in by.items()}}
json.dump(summary, open(f"{ev}/atm_openai_gpt-5-mini_summary.json", "w"), indent=1)
D = json.load(open(f"{ev}/deterministic_accuracy.json")); dby = {}
for r in D: dby.setdefault(r["qtype"], []).append(float(r["accuracy"]))
json.dump({"count": len(D), "accuracy": sum(sum(v) for v in dby.values()) / len(D), "by_qtype": {t: {"count": len(v), "accuracy": sum(v) / len(v)} for t, v in dby.items()}}, open(f"{ev}/deterministic_accuracy_summary.json", "w"), indent=1)
print(NAME, "QS %.2f" % (100 * summary["accuracy"]), {t: round(100 * v["accuracy"], 2) for t, v in summary["by_qtype"].items()}, "n", len(J), "| number answers changed", changed, "net flips", flips)

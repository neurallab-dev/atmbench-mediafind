"""Like merge_eval.py, for any question type: answers (and their judged/deterministic records) of QTYPE come from NEW_RUN, everything
else from BASE_RUN. Writes ans/NAME.jsonl and ans/NAME.eval/ in the official layout. usage: merge_eval_type.py NAME BASE_RUN NEW_RUN QTYPE"""
import json, os, sys, shutil
NAME, BASE, NEW, QT = sys.argv[1:5]
qa = json.load(open("atm/atm-bench.json")); qt = {q["id"]: q["qtype"] for q in qa}
def ans(name): return {json.loads(l)["id"]: json.loads(l) for l in open(f"ans/{name}.jsonl")}
A, N = ans(BASE), ans(NEW)
with open(f"ans/{NAME}.jsonl", "w") as f:
    for q in qa: f.write(json.dumps(N[q["id"]] if qt[q["id"]] == QT else A[q["id"]], ensure_ascii=False) + "\n")
ev = f"ans/{NAME}.eval"; shutil.rmtree(ev, ignore_errors=True); os.makedirs(ev)
def recs(name, fn): return {r["id"]: r for r in json.load(open(f"ans/{name}.eval/{fn}"))}
for fn in ("atm_openai_gpt-5-mini.json", "deterministic_accuracy.json"):
    B, Nr = recs(BASE, fn), recs(NEW, fn)
    json.dump([Nr[q["id"]] if qt[q["id"]] == QT else B[q["id"]] for q in qa], open(f"{ev}/{fn}", "w"), ensure_ascii=False, indent=1)
shutil.copy(f"ans/{BASE}.eval/llm_judge_openai_gpt-5-mini.json", f"{ev}/llm_judge_openai_gpt-5-mini.json")
J = json.load(open(f"{ev}/atm_openai_gpt-5-mini.json")); by = {}
for r in J: by.setdefault(r["qtype"], []).append(float(r["accuracy"]))
summary = {"count": len(J), "correct": sum(sum(v) for v in by.values()), "accuracy": sum(sum(v) for v in by.values()) / len(J),
           "by_qtype": {t: {"count": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)} for t, v in by.items()}}
json.dump(summary, open(f"{ev}/atm_openai_gpt-5-mini_summary.json", "w"), indent=1)
D = json.load(open(f"{ev}/deterministic_accuracy.json")); dby = {}
for r in D: dby.setdefault(r["qtype"], []).append(float(r["accuracy"]))
json.dump({"count": len(D), "accuracy": sum(sum(v) for v in dby.values()) / len(D), "by_qtype": {t: {"count": len(v), "accuracy": sum(v) / len(v)} for t, v in dby.items()}}, open(f"{ev}/deterministic_accuracy_summary.json", "w"), indent=1)
print(NAME, "QS %.2f" % (100 * summary["accuracy"]), {t: round(100 * v["accuracy"], 2) for t, v in summary["by_qtype"].items()}, "n", len(J))

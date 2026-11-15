"""Assemble the type-conditional answer set: number/open_end answers (and their existing gpt-5-mini judgments) from one judged
run, list_recall answers (deterministic Jaccard) from another judged run. Writes ans/NAME.jsonl and ans/NAME.eval/ in the
official evaluator's layout, so make_evidence.py and judge-free re-scoring work on it.
usage: merge_eval.py NAME BASE_RUN LIST_RUN"""
import json, os, sys, shutil
NAME, BASE, LIST = sys.argv[1:4]
qa = json.load(open("atm/atm-bench.json")); qt = {q["id"]: q["qtype"] for q in qa}
def ans(name): return {json.loads(l)["id"]: json.loads(l) for l in open(f"ans/{name}.jsonl")}
A, L = ans(BASE), ans(LIST)
merged = [L[q["id"]] if qt[q["id"]] == "list_recall" else A[q["id"]] for q in qa]
with open(f"ans/{NAME}.jsonl", "w") as f:
    for r in merged: f.write(json.dumps(r, ensure_ascii=False) + "\n")
ev = f"ans/{NAME}.eval"; shutil.rmtree(ev, ignore_errors=True); os.makedirs(ev)
def recs(name, fn): return {r["id"]: r for r in json.load(open(f"ans/{name}.eval/{fn}"))}
for fn in ("atm_openai_gpt-5-mini.json", "deterministic_accuracy.json"):
    B, Lr = recs(BASE, fn), recs(LIST, fn)
    out = [Lr[q["id"]] if qt[q["id"]] == "list_recall" else B[q["id"]] for q in qa if q["id"] in (Lr if qt[q["id"]] == "list_recall" else B)]
    json.dump(out, open(f"{ev}/{fn}", "w"), ensure_ascii=False, indent=1)
shutil.copy(f"ans/{BASE}.eval/llm_judge_openai_gpt-5-mini.json", f"{ev}/llm_judge_openai_gpt-5-mini.json")
J = json.load(open(f"{ev}/atm_openai_gpt-5-mini.json"))
by = {}
for r in J: by.setdefault(r["qtype"], []).append(float(r["accuracy"]))
summary = {"count": len(J), "correct": sum(sum(v) for v in by.values()), "accuracy": sum(sum(v) for v in by.values()) / len(J),
           "by_qtype": {t: {"count": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)} for t, v in by.items()}}
json.dump(summary, open(f"{ev}/atm_openai_gpt-5-mini_summary.json", "w"), indent=1)
D = json.load(open(f"{ev}/deterministic_accuracy.json")); dby = {}
for r in D: dby.setdefault(r["qtype"], []).append(float(r["accuracy"]))
json.dump({"count": len(D), "accuracy": sum(sum(v) for v in dby.values()) / len(D), "by_qtype": {t: {"count": len(v), "accuracy": sum(v) / len(v)} for t, v in dby.items()}}, open(f"{ev}/deterministic_accuracy_summary.json", "w"), indent=1)
print(NAME, "QS %.2f" % (100 * summary["accuracy"]), {t: round(100 * v["accuracy"], 1) for t, v in summary["by_qtype"].items()}, "n", len(J))

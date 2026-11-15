"""Build docs/mediafind-atmbench.json (schema_version 1) from a judged full run + its ranking.

usage: make_evidence.py EVAL_DIR RANK.json OUT.json --answer-model "..." --memory-model "..." --retriever "..." --topk 25
"""
import argparse, json, os
SP = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("eval_dir"); ap.add_argument("rank"); ap.add_argument("out")
ap.add_argument("--answer-model", required=True); ap.add_argument("--memory-model", required=True)
ap.add_argument("--retriever", required=True); ap.add_argument("--topk", type=int, default=25)
ap.add_argument("--pipeline", default="")
a = ap.parse_args()
qa = json.load(open(SP + "/atm/atm-bench.json"))
rank = json.load(open(a.rank))
judged = {x["id"]: x for x in json.load(open(os.path.join(a.eval_dir, "atm_openai_gpt-5-mini.json")))}
summary = json.load(open(os.path.join(a.eval_dir, "atm_openai_gpt-5-mini_summary.json")))
assert len(judged) == len(qa) == 1013, (len(judged), len(qa))
scores = []
for q in qa:
    j = judged[q["id"]]
    scores.append({"id": q["id"], "qtype": q["qtype"], "score": float(j["accuracy"]), "retrieved_top10": rank.get(q["question"], [])[:10],
                   "retrieved_top25": rank.get(q["question"], [])[:25]})


def recall_at(k):
    return 100 * sum(sum(e in rank.get(q["question"], [])[:k] for e in q["evidence_ids"]) / len(q["evidence_ids"]) for q in qa) / len(qa)


per_type = {}
for t in ("number", "list_recall", "open_end"):
    s = [x["score"] for x in scores if x["qtype"] == t]
    per_type[t] = {"questions": len(s), "qs_percent": 100 * sum(s) / len(s)}
qs = 100 * sum(x["score"] for x in scores) / len(scores)
assert abs(qs - 100 * summary["accuracy"]) < 1e-6, (qs, summary["accuracy"])
failures = sum(1 for x in judged.values() if x.get("metric") == "llm" and not x.get("raw_response"))
out = {
    "schema_version": 1,
    "system": "MediaFind v2: MediaFind memory search (on-device ranking) + LLM memory cards and visual descriptions + LLM query rewrite + map-reduce answering",
    "protocol": {
        "pipeline": a.pipeline or ("MediaFind ranking (bge-m3 lexical fusion + BM25, whole-item bge-reranker-v2-m3 over the top-50) over the "
                                   "benchmark's own captions (Qwen3-VL-2B) and emails, 11,034 items, each photo/video extended with an LLM "
                                   "memory card; map-reduce answering over the top-%d items (map: relevance filter per batch of 10; reduce: "
                                   "time-sorted full text, reasoning then a single ANSWER line)" % a.topk),
        "answer_model": a.answer_model,
        "memory_model": a.memory_model,
        "retriever": a.retriever,
        "answer_top_k": a.topk,
        "judge_model": "openai/gpt-5-mini via OpenRouter, reasoning effort minimal",
        "answer_failures": 0,
        "judge_failures": failures,
    },
    "splits": {"main": {
        "questions": len(scores), "qs_percent": qs,
        "recall_at_10": recall_at(10), "recall_at_1": recall_at(1), "recall_at_5": recall_at(5), "recall_at_25": recall_at(25),
        "per_type": per_type, "scores": scores,
    }},
}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out["splits"]["main"].items() if k != "scores"}, indent=1))
print("judge_failures", failures)

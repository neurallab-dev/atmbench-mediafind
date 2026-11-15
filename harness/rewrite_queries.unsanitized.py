"""Rewrite every question into the retrieval query a memory system would use (one flash call each, cached).

usage: rewrite_queries.py OUT.json [model]  -> {qid: rewritten}
Then builds atm/atm-bench-<TAG>-rw.doc.jsonl (env TASKS_TAG=cards2) with the rewritten question text (qids unchanged).
"""
import json, os, sys, re, concurrent.futures as cf
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import llm
out_p = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek/deepseek-v4.1-flash"
TAG = os.environ.get("TASKS_TAG", "cards2")
qa = json.load(open(SP + "/atm/atm-bench.json"))
SYSTEM = ("You turn a person's recall question about their own photos, videos and emails into the search query a memory "
          "system should run. The person is a Cambridge (UK) graduate student who travels to AI conferences (e.g. ACL 2024 "
          "Bangkok Aug 2024, ECCV 2024 Milan Oct 2024, ECAI 2024 Santiago de Compostela Oct 2024, BMVC 2024 Glasgow Nov "
          "2024, NeurIPS 2023 New Orleans Dec 2023, NAACL 2025 Albuquerque May 2025, WWW 2025 Sydney Apr 2025) and on holidays.")
INSTR = ("Rewrite the question below as ONE retrieval query: keep every original term that could appear in the item text "
         "(names, dishes, objects, places, dates), expand event names into their place and dates, name the likely item kind "
         "(photo, video, email: hotel booking, receipt, ticket, invitation), resolve relative dates using any 'Today is …' "
         "in the question into absolute month/year, and add 3-6 synonyms a caption might use instead (e.g. 'camouflaged car, "
         "test paint wrap'). Output only the query, one line, no quotes, under 60 words.")


def rw(q):
    try:
        out = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{INSTR}\n\nQuestion: {q['question']}"}],
                       MODEL, max_tokens=200, tag="rw:" + q["id"])
        return q["id"], " ".join(out.strip().split())
    except Exception as e:
        print("FAILED", q["id"], repr(e)[:120], file=sys.stderr, flush=True)
        return q["id"], q["question"]


with cf.ThreadPoolExecutor(16) as ex:
    rew = dict(ex.map(rw, qa))
json.dump(rew, open(out_p, "w"), ensure_ascii=False, indent=0)
print("rewrites", len(rew), "|", llm.cost_line())
for q in qa[:4]:
    print("Q:", q["question"][:90]); print("->", rew[q["id"]][:160])
task = json.loads(open(os.path.join(SP, "atm", f"atm-bench-{TAG}.doc.jsonl")).readline())
seen = set()
for qq in task["questions"]:
    r = rew[qq["qid"]]
    if r in seen or r == qq["question"]:   # keep keys unique: append the qid tail
        r = r + " [" + qq["qid"][:8] + "]"
    seen.add(r); qq["question"] = r
with open(os.path.join(SP, "atm", f"atm-bench-{TAG}-rw.doc.jsonl"), "w") as f:
    f.write(json.dumps(task, ensure_ascii=False) + "\n")
json.dump({qq["question"]: qq["qid"] for qq in task["questions"]}, open(os.path.join(SP, "atm", f"rw-{TAG}-key.json"), "w"), ensure_ascii=False)
print("wrote rewritten tasks file and key map")

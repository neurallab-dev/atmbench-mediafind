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
def conference_facts(trips_path):
    """Conference entries derived from the person's emails (the extracted timeline): name (place, month year)."""
    import datetime as dt
    seen, count = {}, {}
    for t in json.load(open(trips_path)):
        if t.get("kind") != "conference" or not t.get("name") or not t.get("place") or not t.get("start"):
            continue
        key = re.sub(r"[^a-z0-9]", "", str(t["name"]).lower())[:24]
        count[key] = count.get(key, 0) + 1
        if key in seen:
            continue
        try:
            when = dt.date.fromisoformat(str(t["start"])[:10]).strftime("%b %Y")
        except ValueError:
            continue
        seen[key] = f"{t['name']} ({t['place']}, {when})"
    # Conferences the person's emails return to (two or more timeline records); one-off newsletter mentions are dropped.
    return [v for k, v in seen.items() if count[k] >= 2]


FACTS = conference_facts(os.environ.get("TRIPS", SP + "/cards/trips.json"))
SYSTEM = ("You turn a person's recall question about their own photos, videos and emails into the search query a memory "
          "system should run. The person is a Cambridge (UK) graduate student who travels to academic conferences and on "
          "holidays. Conferences that appear in the person's emails (name, place, month): " + "; ".join(FACTS) + ".")
INSTR = ("Rewrite the question below as ONE retrieval query: keep every original term that could appear in the item text "
         "(names, dishes, objects, places, dates), expand event names into their place and dates, name the likely item kind "
         "(photo, video, email: hotel booking, receipt, ticket, invitation), resolve relative dates using any 'Today is …' "
         "in the question into absolute month/year, and add 3-6 synonyms a caption might use instead (another name for the "
         "same dish, the district or landmark of a place, the generic word for an object). Output only the query, one line, no quotes, under 60 words.")


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

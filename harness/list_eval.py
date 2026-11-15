"""Post-hoc list_recall comparison (deterministic Jaccard, no judge): the reduce's answer vs the map-kept IDs.

usage: [BEST_GUESS=1 MAP_MODEL=... ANSWER_MODEL=... ITEMS_FILE=...] list_eval.py ANSWERS.jsonl RANK.json [topk] [EVAL_DIR]
Map calls are cached, so this costs nothing. If EVAL_DIR is given, the official per-question accuracy for the
reduce answers is printed beside the local Jaccard to confirm the scoring matches.
"""
import json, os, re, sys
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import mf_answer_v2 as m

ans_p, rank_p = sys.argv[1], sys.argv[2]
topk = int(sys.argv[3]) if len(sys.argv) > 3 else 25
eval_dir = sys.argv[4] if len(sys.argv) > 4 else None
qa = [q for q in json.load(open(SP + "/atm/atm-bench.json")) if q["qtype"] == "list_recall"]
rank = json.load(open(rank_p))
answers = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(ans_p)}
official = {}
if eval_dir:
    official = {x["id"]: float(x["accuracy"]) for x in json.load(open(os.path.join(eval_dir, "atm_openai_gpt-5-mini.json")))}
ID = re.compile(r"\b(?:email\d{12}|\d{8}_\d{6})\b")


def ids(s):
    return set(ID.findall(s))


def jaccard(pred, gold):
    p, g = ids(pred), ids(gold)
    return len(p & g) / len(p | g) if (p | g) else 0.0


def wanted_types(q):
    ql = q.lower()
    t = set()
    if re.search(r"\b(photo|photos|picture|pictures|image|images|shot|shots)\b", ql): t.add("image")
    if re.search(r"\b(video|videos|footage|clip|clips|recording)\b", ql): t.add("video")
    if re.search(r"\b(email|emails|e-mail|mail)\b", ql): t.add("email")
    return t


tot = {"reduce": 0.0, "map": 0.0, "map_typed": 0.0, "official": 0.0}
rows = []
for q in qa:
    pool = [i for i in rank.get(q["question"], [])[:topk] if i in m.ITEMS]
    kept = []
    for s in range(0, len(pool), 10):
        kept += m.map_batch(q["question"], pool[s:s + 10])
    kept_ids = sorted(dict.fromkeys(i for i, _ in kept), key=m.ts)
    want = wanted_types(q["question"])
    typed = [i for i in kept_ids if not want or m.ITEMS[i]["modality"] in want] or kept_ids
    a_red = answers.get(q["id"], "")
    sc = {"reduce": jaccard(a_red, q["answer"]), "map": jaccard(", ".join(kept_ids), q["answer"]),
          "map_typed": jaccard(", ".join(typed), q["answer"]), "official": official.get(q["id"], float("nan"))}
    for k in tot: tot[k] += 0 if sc[k] != sc[k] else sc[k]
    rows.append((q["id"], sc, len(kept_ids), len(typed), q["question"][:70]))
n = len(qa)
print(f"list_recall n={n}: reduce {100*tot['reduce']/n:.2f}  map-only {100*tot['map']/n:.2f}  map-typed {100*tot['map_typed']/n:.2f}"
      + (f"  (official reduce {100*tot['official']/n:.2f})" if official else ""))
diff = [r for r in rows if abs(r[1]["map_typed"] - r[1]["reduce"]) > 0.3]
for r in diff[:20]:
    print(f"  {r[1]['reduce']:.2f} -> typed {r[1]['map_typed']:.2f} (kept {r[2]}, typed {r[3]}) {r[4]}")
print("local jaccard == official on reduce answers:", sum(1 for r in rows if official and abs(r[1]['reduce'] - r[1]['official']) < 1e-6), "/", n if official else "n/a")

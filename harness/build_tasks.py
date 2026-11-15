"""Build card-augmented tasks file + items file.

usage: build_tasks.py CARDS.jsonl OUT_TAG [position]
  position: head (default; card inserted after the 'Short Caption:' line) | tail (appended at the end)
Writes atm/atm-bench-<TAG>.doc.jsonl and atm/items-<TAG>.json
"""
import json, re, sys, os
SP = os.path.dirname(os.path.abspath(__file__))
cards_p, tag = sys.argv[1], sys.argv[2]
pos = sys.argv[3] if len(sys.argv) > 3 else "head"
vision_p = sys.argv[4] if len(sys.argv) > 4 else None
cards = {}
for line in open(cards_p):
    r = json.loads(line); cards[r["item_id"]] = r["card"].strip()
print(len(cards), "cards")
if vision_p:
    nv = 0
    for line in open(vision_p):
        r = json.loads(line)
        if r.get("desc") and r["item_id"] in cards:
            cards[r["item_id"]] += "\nVisual detail: " + r["desc"].strip(); nv += 1
    print(nv, "visual descriptions appended")


def augment(text, card):
    block = "Memory card:\n" + card
    if pos == "tail":
        return text.rstrip() + "\n" + block
    m = re.search(r"^Short Caption: .*$", text, flags=re.M)
    if not m:
        return text.rstrip() + "\n" + block
    return text[: m.end()] + "\n" + block + text[m.end():]


task = json.loads(open(os.path.join(SP, "atm", "atm-bench.doc.jsonl")).readline())
k = 0
for s in task["sessions"]:
    c = cards.get(s["session_id"])
    if c:
        for t in s["turns"]:
            t["text"] = augment(t["text"], c); k += 1
out = os.path.join(SP, "atm", f"atm-bench-{tag}.doc.jsonl")
with open(out, "w") as f:
    f.write(json.dumps(task, ensure_ascii=False) + "\n")
print(out, "sessions", len(task["sessions"]), "augmented", k, "questions", len(task["questions"]))
items = json.load(open(os.path.join(SP, "atm", "items.json")))
n = 0
for i in items:
    c = cards.get(i["item_id"])
    if c:
        i["text"] = augment(i["text"], c); n += 1
json.dump(items, open(os.path.join(SP, "atm", f"items-{tag}.json"), "w"), ensure_ascii=False)
print(f"items-{tag}.json augmented {n}")

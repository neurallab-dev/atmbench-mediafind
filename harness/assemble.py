"""Concatenate the three per-type answer files into one prediction file in the order of atm/atm-bench.json and print
its SHA-256.  usage: assemble.py OUT.jsonl NUMBER.jsonl LIST.jsonl OPEN.jsonl"""
import hashlib, json, os, sys
SP = os.path.dirname(os.path.abspath(__file__))
out_p, parts = sys.argv[1], sys.argv[2:]
qa = json.load(open(SP + "/atm/atm-bench.json")); want = {"number": parts[0], "list_recall": parts[1], "open_end": parts[2]}
ans = {}
for t, p in want.items():
    for line in open(p):
        r = json.loads(line); ans[r["id"]] = (t, r["answer"])
missing = [q["id"] for q in qa if q["id"] not in ans]
wrong_type = [q["id"] for q in qa if q["id"] in ans and ans[q["id"]][0] != q["qtype"]]
empty = [q["id"] for q in qa if q["id"] in ans and not str(ans[q["id"]][1]).strip()]
assert not missing and not wrong_type, (len(missing), len(wrong_type))
with open(out_p, "w") as f:
    for q in qa:
        f.write(json.dumps({"id": q["id"], "answer": ans[q["id"]][1]}, ensure_ascii=False) + "\n")
print(len(qa), "predictions;", len(empty), "empty answers;", "sha256", hashlib.sha256(open(out_p, "rb").read()).hexdigest())

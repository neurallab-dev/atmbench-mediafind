"""Per-type question files for the re-run: atm/atm-bench-{number,list-only,open}.json (filtered from atm/atm-bench.json)."""
import json, os
SP = os.path.dirname(os.path.abspath(__file__))
qa = json.load(open(SP + "/atm/atm-bench.json"))
for t, name in (("number", "atm-bench-number.json"), ("list_recall", "atm-bench-list-only.json"), ("open_end", "atm-bench-open.json")):
    sub = [q for q in qa if q["qtype"] == t]
    json.dump(sub, open(os.path.join(SP, "atm", name), "w"), ensure_ascii=False)
    print(name, len(sub))

"""Update the existing 'MediaFind v2' row's numbers and evidence file in leaderboard.html from a new evidence json.

usage: update_board_v2.py FORK_DIR EVIDENCE.json   (branch already carries the v2 row and § note)
"""
import json, re, sys, shutil, os
fork, ev_p = sys.argv[1], sys.argv[2]
ev = json.load(open(ev_p)); m = ev["splits"]["main"]
qs, r10 = round(m["qs_percent"], 2), round(m["recall_at_10"], 2)
p = os.path.join(fork, "leaderboard.html"); s = open(p, encoding="utf-8").read()
row_re = re.compile(r"(\{ type: 'Memory', harness: 'MediaFind v2',[^\n]*?qs: )([0-9.]+)(, recall: )([0-9.]+)(,)")
assert len(row_re.findall(s)) == 1, "expected exactly one MediaFind v2 row"
old = row_re.search(s)
s = row_re.sub(lambda mm: f"{mm.group(1)}{qs:.2f}{mm.group(3)}{r10:.2f}{mm.group(5)}", s)
open(p, "w", encoding="utf-8").write(s)
shutil.copy(ev_p, os.path.join(fork, "docs", "mediafind-v2-atmbench.json"))
print(f"v2 row: qs {old.group(2)} -> {qs:.2f}, recall {old.group(4)} -> {r10:.2f}; evidence replaced")

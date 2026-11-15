"""Check recorded `grep -n` lines and `sed -n A,Bp` blocks against the rebuilt version of the file at that moment."""
import json, os, re, shlex, collections, bisect
ROOT = os.path.expanduser("~/.claude/projects"); HOME = "/Users/mbp"
REAL = HOME + "/.cache/mediafind-eval/atm-first"; HERE = os.path.abspath(".")
TX = [l.strip() for l in open("tx_list.txt") if l.strip()]
hist = collections.defaultdict(list)
for l in open("versions/index.jsonl"):
    x = json.loads(l); hist[x["rel"]].append((x["ts"], x["sha1"]))
def version_at(rel, ts):
    h = hist.get(rel)
    if not h: return None
    i = bisect.bisect_right([t for t, _ in h], ts) - 1
    # a version recorded at the call's own timestamp was produced by that same call: skip it
    if i < 0: return None
    return open(os.path.join("versions", h[i][1])).read().split("\n")
def norm(p):
    p = p.strip("\"'")
    if p.startswith("~/"): p = HOME + p[1:]
    return os.path.normpath(p.replace("$HOME", HOME))
stats = collections.defaultdict(lambda: [0, 0]); bad = []
for rel in TX:
    calls, outs = {}, {}
    for line in open(os.path.join(ROOT, rel), errors="replace"):
        try: o = json.loads(line)
        except Exception: continue
        m = o.get("message", {}) or {}
        for c in (m.get("content") if isinstance(m.get("content"), list) else []):
            if not isinstance(c, dict): continue
            if c.get("type") == "tool_use" and c.get("name") == "Bash": calls[c["id"]] = (o.get("timestamp", ""), c["input"].get("command", ""))
            elif c.get("type") == "tool_result":
                cc = c.get("content"); cc = "".join(b.get("text", "") for b in cc if isinstance(b, dict)) if isinstance(cc, list) else (cc or "")
                outs[c.get("tool_use_id")] = cc
    for cid, (ts, cmd) in calls.items():
        out = outs.get(cid)
        if out is None or "atm-first" not in cmd: continue
        env = dict(re.findall(r"(?:^|[;&\n]\s*)([A-Za-z_]\w*)=([^\s;&|]+)", cmd))
        c2 = cmd
        for k, v in sorted(env.items(), key=lambda kv: -len(kv[0])): c2 = c2.replace("${%s}" % k, v).replace("$" + k, v)
        cds = re.findall(r"(?:^|[;&\n(]\s*)cd\s+(\S+)", c2); cwd = norm(cds[0]) if len(cds) == 1 else None
        def rel_of(a):
            p = norm(a) if a.startswith(("/", "~")) else (os.path.join(cwd, a) if cwd else None)
            return p[len(REAL):] if p and p.startswith(REAL + "/") else None
        # (a) grep -n on exactly one harness file, and the only line-numbered producer in the command
        greps = re.findall(r"\bgrep\s+((?:-[A-Za-z]+\s+)*)-n\w*\s+((?:-[A-Za-z]+\s+)*)('[^']*'|\"[^\"]*\"|\S+)\s+(\S+\.(?:py|sh))\b(?!\s+\S+\.(?:py|sh))", c2)
        if len(greps) == 1 and len(re.findall(r"\b(grep|nl|cat -n)\b[^|;&]*-n", c2)) == 1 and "sed -n" not in c2:
            r_ = rel_of(greps[0][3]); v = version_at(r_, ts) if r_ else None
            if v:
                for ln in out.split("\n"):
                    m = re.match(r"^(\d+):(.*)$", ln)
                    if not m: continue
                    n, text = int(m.group(1)), m.group(2)
                    ok = 0 < n <= len(v) and v[n - 1] == text
                    if len(text) > 2000: continue
                    stats[r_][0 if ok else 1] += 1
                    if not ok: bad.append((ts, r_, n, text[:160], (v[n - 1][:160] if 0 < n <= len(v) else "<eof>")))
        # (b) sed -n 'A,Bp' FILE, one occurrence
        seds = re.findall(r"\bsed\s+-n\s+'?(\d+),(\d+)p'?\s+(\S+\.(?:py|sh))\b", c2)
        if len(seds) == 1:
            a, b, f = seds[0]; r_ = rel_of(f); v = version_at(r_, ts) if r_ else None
            if v:
                block = "\n".join(v[int(a) - 1:int(b)]).rstrip()
                ok = block and block in out
                stats[r_][0 if ok else 1] += 1
                if not ok: bad.append((ts, r_, f"sed {a},{b}", "", ""))
for k, (a, b) in sorted(stats.items()): print(f"{k:32s} ok {a:4d}  bad {b:3d}")
json.dump(bad, open("verify2_bad.json", "w"), indent=1); print("bad:", len(bad))

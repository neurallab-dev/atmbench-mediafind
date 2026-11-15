"""Re-run read-only commands that old sessions ran on harness files (after each file's last change) against
./tree and compare with the recorded tool output."""
import json, os, re, shlex, subprocess, collections
ROOT = os.path.expanduser("~/.claude/projects")
HOME = "/Users/mbp"; REAL = HOME + "/.cache/mediafind-eval/atm-first"; TREE = os.path.abspath("tree")
TX = [l.strip() for l in open("tx_list.txt") if l.strip()]
# last modification time per tree file
lastmod = {}
for l in open("replay_log.jsonl"):
    x = json.loads(l)
    for k in ("target", "dst"):
        if x.get(k) and x.get("ok", True) and x.get("rc", 0) == 0: lastmod[x[k]] = x["ts"]
    for f in x.get("files", []) or []: lastmod[f] = x["ts"]
    if x["op"] == "py" and x["rc"] == 0:
        for m in re.findall(r"open\(\s*[\"']([\w./-]+\.(?:py|sh))[\"']\s*,\s*[\"']w", x["snippet"]):
            p = m if m.startswith("/") else "/" + m
            lastmod[p] = x["ts"]
READ = re.compile(r"^\s*(grep|sed\s+-n|head|tail|wc|cat)\b")
def norm(p):
    p = p.strip("\"'")
    if p.startswith("~/"): p = HOME + p[1:]
    return os.path.normpath(p.replace("$HOME", HOME))
res = collections.defaultdict(lambda: [0, 0]); bad = []
for rel in TX:
    calls, outs = {}, {}
    for line in open(os.path.join(ROOT, rel), errors="replace"):
        try: o = json.loads(line)
        except Exception: continue
        msg = o.get("message", {}) or {}
        for c in (msg.get("content") or []) if isinstance(msg.get("content"), list) else []:
            if not isinstance(c, dict): continue
            if c.get("type") == "tool_use" and c.get("name") == "Bash":
                calls[c["id"]] = (o.get("timestamp", ""), c["input"].get("command", ""))
            elif c.get("type") == "tool_result":
                cc = c.get("content")
                if isinstance(cc, list): cc = "".join(b.get("text", "") for b in cc if isinstance(b, dict))
                outs[c.get("tool_use_id")] = cc or ""
    for cid, (ts, cmd) in calls.items():
        if cid not in outs or "atm-first" not in cmd: continue
        env = dict(re.findall(r"(?:^|[;&\n]\s*)([A-Za-z_]\w*)=([^\s;&|]+)", cmd))
        c2 = cmd
        for k, v in sorted(env.items(), key=lambda kv: -len(kv[0])):
            c2 = c2.replace("${%s}" % k, v).replace("$" + k, v)
        segs = re.split(r"\s*(?:&&|;|\n)\s*", c2.strip())
        cwd = None; runs = []; ok = True
        for sg in segs:
            if not sg: continue
            if re.match(r"^\s*([A-Za-z_]\w*)=\S+$", sg): continue
            m = re.match(r"^\s*cd\s+(\S+)$", sg)
            if m: cwd = norm(m.group(1)); continue
            if not READ.match(sg) or re.search(r"[|<>`$]|\*", sg): ok = False; break
            runs.append(sg)
        if not ok or not runs: continue
        # every file arg must map into the tree and be a harness file whose last change is before ts
        files_ok = True; mapped = []
        for sg in runs:
            try: w = shlex.split(sg)
            except ValueError: files_ok = False; break
            fargs = [a for a in w[1:] if re.search(r"\.(py|sh)$", a)]
            if not fargs: files_ok = False; break
            new = [w[0]]
            for a in w[1:]:
                if re.search(r"\.(py|sh)$", a):
                    p = norm(a) if a.startswith(("/", "~", "$")) else (os.path.join(cwd, a) if cwd else None)
                    if not p or not p.startswith(REAL + "/"): files_ok = False; break
                    tp = TREE + p[len(REAL):]
                    key = p[len(REAL):]
                    if not os.path.exists(tp) or lastmod.get(key, "") > ts: files_ok = False; break
                    new.append(tp)
                else: new.append(a)
            if not files_ok: break
            mapped.append((sg, new, [a for a in w[1:] if re.search(r"\.(py|sh)$", a)]))
        if not files_ok: continue
        got = ""
        for sg, argv, _ in mapped:
            r = subprocess.run(argv, capture_output=True, text=True); got += r.stdout + r.stderr
        want = re.sub(r"\n?Shell cwd was reset to .*$", "", outs[cid]).rstrip()
        got = got.replace(TREE + "/", REAL + "/").replace(TREE, REAL).rstrip()
        # grep with several files prefixes the path; normalize both
        same = got == want or got.replace(REAL + "/", "") == want.replace(REAL + "/", "")
        for _, _, fa in mapped:
            for f in fa: res[os.path.basename(f)][0 if same else 1] += 1
        if not same: bad.append((ts, cmd[:200], want[:400], got[:400]))
for k, (a, b) in sorted(res.items()): print(f"{k:30s} match {a:3d}  mismatch {b:3d}")
json.dump(bad, open("verify_mismatches.json", "w"), indent=1)
print("mismatches:", len(bad))

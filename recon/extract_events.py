"""Collect every file-mutating tool call that touched the ATM harness, from Claude session transcripts.

Output: events.jsonl, one event per line, sorted by timestamp:
  {ts, session, kind: write|edit|heredoc|append|unparsed, path, key, content|old|new|replace_all|cmd}
key = path relative to the ATM root (atm-first/ or an ATM session scratchpad/), else basename.
"""
import json, os, re, sys

ROOT = os.path.expanduser("~/.claude/projects")
TX = [l.strip() for l in open("tx_list.txt") if l.strip()]
HOME = os.path.expanduser("~")

ATM_ROOT_RE = re.compile(r"(?:/Users/mbp|~|\$HOME)/\.cache/mediafind-eval/atm-first/")
SCRATCH_RE = re.compile(r"/private/tmp/claude-501/[^/]+/(?:4471c2e1|0d009b2d)[^/]*/scratchpad/")

def key_of(path):
    p = path.replace("~/", HOME + "/").replace("$HOME/", HOME + "/")
    m = ATM_ROOT_RE.search(p.replace(HOME, "/Users/mbp"))
    if m:
        return "atm:" + p.replace(HOME, "/Users/mbp")[m.end():]
    m = SCRATCH_RE.search(p)
    if m:
        return "scr:" + p[m.end():]
    return None

HEREDOC = re.compile(r"""(?:^|[;&|\n]\s*|\(\s*)(?:cat|tee)\s*(>>?)?\s*(?:-a\s+)?([^\s<;&|]+)\s*<<-?\s*(['"]?)(\w+)\3[^\n]*\n(.*?)\n\4(?=\n|$|\s|\))""", re.S)
CD = re.compile(r"(?:^|[;&|\n]\s*|\(\s*)cd\s+([^\s;&|)]+)")
VARDEF = re.compile(r"(?:^|[;&|\n]\s*|export\s+)(\w+)=([^\s;&|]+)")

def resolve(fname, cmd_before, env):
    if fname.startswith(("/", "~", "$")):
        p = fname
    else:
        cds = CD.findall(cmd_before)
        base = cds[-1] if cds else ""
        p = (base.rstrip("/") + "/" + fname) if base else fname
    for k, v in sorted(env.items(), key=lambda kv: -len(kv[0])):
        p = p.replace("${%s}" % k, v).replace("$" + k, v)
    p = p.replace('"', "").replace("'", "")
    return p

events = []
for rel in TX:
    fn = os.path.join(ROOT, rel)
    sess = os.path.basename(fn)[:8]
    for line in open(fn, errors="replace"):
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("type") != "assistant":
            continue
        ts = o.get("timestamp", "")
        for c in (o.get("message", {}).get("content") or []):
            if not isinstance(c, dict) or c.get("type") != "tool_use":
                continue
            name, inp = c.get("name"), c.get("input") or {}
            if name == "Write":
                k = key_of(inp.get("file_path", ""))
                if k:
                    events.append(dict(ts=ts, session=sess, kind="write", path=inp["file_path"], key=k, content=inp.get("content", "")))
            elif name == "Edit":
                k = key_of(inp.get("file_path", ""))
                if k:
                    events.append(dict(ts=ts, session=sess, kind="edit", path=inp["file_path"], key=k, old=inp.get("old_string", ""), new=inp.get("new_string", ""), replace_all=bool(inp.get("replace_all"))))
            elif name == "MultiEdit":
                k = key_of(inp.get("file_path", ""))
                if k:
                    for e in inp.get("edits", []):
                        events.append(dict(ts=ts, session=sess, kind="edit", path=inp["file_path"], key=k, old=e.get("old_string", ""), new=e.get("new_string", ""), replace_all=bool(e.get("replace_all"))))
            elif name == "Bash":
                cmd = inp.get("command", "")
                if "atm-first" not in cmd and "scratchpad" not in cmd and "$SP" not in cmd:
                    continue
                env = {"HOME": "/Users/mbp"}
                for k_, v_ in VARDEF.findall(cmd):
                    env.setdefault(k_, v_.strip('"\''))
                if "SP" not in env and "atm-first" in cmd:
                    env["SP"] = "/Users/mbp/.cache/mediafind-eval/atm-first"
                covered = []
                for m in HEREDOC.finditer(cmd):
                    op, fname, _, tag, body = m.groups()
                    if fname in ("/dev/null", "-"):
                        continue
                    p = resolve(fname, cmd[: m.start()], env)
                    k = key_of(p)
                    if not k and not p.startswith("/") and "atm-first" in cmd[: m.start()]:
                        k = "atm:" + p  # relative to an unparsed cd into atm-first
                    if k:
                        events.append(dict(ts=ts, session=sess, kind="append" if op == ">>" else "heredoc", path=p, key=k, content=body + "\n"))
                        covered.append((m.start(), m.end()))
                # other mutations we cannot replay exactly
                rest = cmd
                for a, b in reversed(covered):
                    rest = rest[:a] + rest[b:]
                if re.search(r"\bsed\s+-i|\bperl\s+-pi|\bpython3?\s+-\s*<<|\bcp\s|\bmv\s|>\s*[\w./$~-]+\.(py|sh|json|md|txt)\b|\bpatch\b|\.write_text\(|open\([^)]*['\"]w", rest):
                    events.append(dict(ts=ts, session=sess, kind="unparsed", path="", key="?", cmd=rest[:6000]))

events.sort(key=lambda e: e["ts"])
with open("events.jsonl", "w") as f:
    for e in events:
        f.write(json.dumps(e) + "\n")
from collections import Counter
print("events", len(events), Counter(e["kind"] for e in events))

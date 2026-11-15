"""Replay harness file history into ./tree (a mirror of ~/.cache/mediafind-eval/atm-first).

Applies, in timestamp order: Write / heredoc writes and appends, and from other Bash commands the
mutation fragments only: `sed -i`, `cp`, `mv`, and inline Python heredocs that write files. Python
fragments run sandboxed: atm-first paths are rewritten to ./tree, writes elsewhere go to /dev/null,
sockets and subprocesses raise. Results: tree/, replay_log.jsonl (one line per fragment and outcome).
"""
import json, os, re, shlex, shutil, subprocess, sys, tempfile

HOME = "/Users/mbp"
REAL = HOME + "/.cache/mediafind-eval/atm-first"
HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.join(HERE, "tree")
LOG = open(os.path.join(HERE, "replay_log.jsonl"), "w")
shutil.rmtree(TREE, ignore_errors=True); os.makedirs(TREE)

def norm(p):
    p = p.strip().strip('"').strip("'")
    if p.startswith("~/"): p = HOME + p[1:]
    p = p.replace("$HOME", HOME).replace("${HOME}", HOME)
    return os.path.normpath(p)

def to_tree(p):
    p = norm(p)
    if p == REAL or p.startswith(REAL + "/"):
        return TREE + p[len(REAL):]
    return None

def log(**k): LOG.write(json.dumps(k) + "\n")

PREAMBLE = r'''
import builtins, os, socket, subprocess, pathlib, io
_TREE = %r
def _ok(p):
    try: p = os.path.abspath(os.fspath(p))
    except Exception: return False
    return p.startswith(_TREE)
_open = builtins.open
def _guard_open(file, mode="r", *a, **k):
    if any(c in mode for c in "wax+") and not _ok(file):
        return _open(os.devnull, mode.replace("x","w").replace("a","w"), *a, **k)
    return _open(file, mode, *a, **k)
builtins.open = _guard_open
io.open = _guard_open
_pw = pathlib.Path.write_text
def _wt(self, *a, **k):
    if not _ok(self): return 0
    return _pw(self, *a, **k)
pathlib.Path.write_text = _wt
for _n in ("remove","unlink","rename","replace","rmdir","removedirs"):
    _f = getattr(os, _n)
    def _mk(f):
        def g(p, *a, **k):
            if not _ok(p): return None
            return f(p, *a, **k)
        return g
    setattr(os, _n, _mk(_f))
def _no(*a, **k): raise RuntimeError("blocked in replay")
socket.socket = _no; socket.create_connection = _no
subprocess.run = _no; subprocess.Popen = _no; subprocess.call = _no; subprocess.check_output = _no; os.system = _no
'''

VARDEF = re.compile(r"(?:^|[;&|\n]\s*|export\s+)([A-Za-z_]\w*)=(\"[^\"]*\"|'[^']*'|[^\s;&|]+)")
FRAG = re.compile(
    r"(?P<py>(?:/usr/bin/)?(?:[\w./-]*python3?|[\w./-]*/python)\s+(?:-I\s+)?-\s*(?:[^<\n]*?)<<-?\s*(['\"]?)(?P<pytag>\w+)\2[^\n]*\n(?P<pybody>.*?)\n(?P=pytag)(?=\s|$|\)))"
    r"|(?P<hd>(?:cat|tee)\s*(?P<hdop>>>?)?\s*(?P<hdf>[^\s<;&|]+)\s*<<-?\s*(['\"]?)(?P<hdtag>\w+)\7[^\n]*\n(?P<hdbody>.*?)\n(?P=hdtag)(?=\s|$|\)))"
    r"|(?P<sed>\bsed\s+-i\s*(?:''|\"\")?(?=\s))"
    r"|(?P<sedr>\bsed\s+(?!-i)(?=['\"-]))"
    r"|(?P<cp>\bcp\s+)"
    r"|(?P<mv>\bmv\s+)"
    r"|(?P<cd>(?:^|(?<=[;&|\n(]))\s*cd\s+(?P<cddir>[^\s;&|)]+))",
    re.S)

def words_from(cmd, pos):
    lx = shlex.shlex(cmd[pos:], posix=True, punctuation_chars=";&|<>")
    lx.whitespace_split = True
    out = []
    try:
        for t in lx:
            if t and set(t) <= set(";&|<>"):
                break
            out.append(t)
    except ValueError:
        return None
    return out

def subst(s, env):
    for k, v in sorted(env.items(), key=lambda kv: -len(kv[0])):
        s = s.replace("${%s}" % k, v).replace("$" + k, v)
    return s

def resolve(p, cwd, env):
    p = subst(p, env).strip().strip('"').strip("'")
    if not p.startswith(("/", "~")):
        p = os.path.join(cwd, p)
    return norm(p)

def run_py(body, cwd, ts):
    real_cwd = cwd
    tcwd = to_tree(real_cwd) or TREE
    os.makedirs(tcwd, exist_ok=True)
    b = body.replace(REAL, TREE).replace("~/.cache/mediafind-eval/atm-first", TREE)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=HERE) as f:
        f.write(PREAMBLE % TREE + "\nimport os\nos.chdir(%r)\n" % tcwd + b)
        fn = f.name
    try:
        r = subprocess.run([sys.executable, "-I", fn], capture_output=True, text=True, timeout=60, cwd=tcwd,
                           env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"})
        return r.returncode, (r.stdout[-300:] + r.stderr[-600:])
    except subprocess.TimeoutExpired:
        return -9, "timeout"
    finally:
        os.unlink(fn)

import hashlib
VERS = os.path.join(HERE, "versions"); shutil.rmtree(VERS, ignore_errors=True); os.makedirs(VERS)
_hash = {}
def snapshot(ts):
    for dp, _, fns in os.walk(TREE):
        for fn in fns:
            fp = os.path.join(dp, fn); rel = fp[len(TREE):]
            if not re.search(r"\.(py|sh|md|txt)$", fn): continue
            h = hashlib.sha1(open(fp, "rb").read()).hexdigest()
            if _hash.get(rel) != h:
                _hash[rel] = h
                with open(os.path.join(VERS, "index.jsonl"), "a") as f:
                    f.write(json.dumps({"ts": ts, "rel": rel, "sha1": h}) + "\n")
                shutil.copy2(fp, os.path.join(VERS, h))
ev = [json.loads(l) for l in open(os.path.join(HERE, "events.jsonl"))]
_prev_ts = None
for e in ev:
    ts = e["ts"]
    if _prev_ts is not None: snapshot(_prev_ts)
    _prev_ts = ts
    if e["kind"] in ("write", "heredoc", "append"):
        if not e["key"].startswith("atm:"): continue
        tp = to_tree(e["path"])
        if not tp:
            rel = e["key"][4:]
            if rel.startswith("/") or ".." in rel or "$" in rel: continue
            tp = os.path.join(TREE, rel)
        os.makedirs(os.path.dirname(tp), exist_ok=True)
        with open(tp, "a" if e["kind"] == "append" else "w") as f: f.write(e["content"])
        log(ts=ts, op=e["kind"], target=tp[len(TREE):], ok=True)
        continue
    cmd = e["cmd"]
    if REAL not in cmd and "atm-first" not in cmd: continue
    env = {"HOME": HOME}
    for k, v in VARDEF.findall(cmd):
        env.setdefault(k, norm(v) if v.strip("\"'").startswith(("/", "~")) else v.strip("\"'"))
    cwd = "/nonexistent"
    for m in FRAG.finditer(cmd):
        if m.group("cd"):
            cwd = resolve(m.group("cddir"), cwd, env); continue
        if m.group("py"):
            body = subst(m.group("pybody"), env) if m.group("pytag") and m.group(2) != "'" else m.group("pybody")
            writes = re.search(r"open\([^)]*['\"][wa]|write_text|\.write\(|os\.(rename|replace|remove)|shutil\.", body)
            if not writes: continue
            if REAL not in body and "atm-first" not in body and not to_tree(cwd): continue
            if not re.search(r"\.(py|sh|md|txt)\b", body): continue
            rc, out = run_py(body, cwd, ts)
            log(ts=ts, op="py", cwd=cwd, rc=rc, out=out, snippet=body[:400])
        elif m.group("hd"):
            pass  # already captured by extract_events (heredoc events)
        elif m.group("sed"):
            args = words_from(cmd, m.end())
            if args is None: log(ts=ts, op="sed", ok=False, why="shlex", raw=cmd[m.start():m.start()+300]); continue
            files = [a for a in args if not a.startswith("-") and (to_tree(resolve(a, cwd, env)) and os.path.exists(to_tree(resolve(a, cwd, env))))]
            exprs = [a for a in args if a not in files]
            if not files: continue
            tfiles = [to_tree(resolve(a, cwd, env)) for a in files]
            r = subprocess.run(["sed", "-i", ""] + exprs + tfiles, capture_output=True, text=True)
            log(ts=ts, op="sed", files=[t[len(TREE):] for t in tfiles], rc=r.returncode, err=r.stderr[:300], expr=" ".join(exprs)[:300])
        elif m.group("sedr"):
            lx = shlex.shlex(cmd[m.end():], posix=True, punctuation_chars=";&|<>")
            lx.whitespace_split = True
            toks = []
            try:
                for t in lx:
                    toks.append(t)
                    if len(toks) > 400: break
            except ValueError:
                continue
            if ">" not in toks: continue
            i = toks.index(">")
            if any(set(t) <= set(";&|<") and t for t in toks[:i]): continue
            args, dst = list(toks[:i]), resolve(toks[i + 1], cwd, env)
            src = resolve(args[-1], cwd, env); tsrc, tdst = to_tree(src), to_tree(dst)
            if not (tsrc and tdst and os.path.isfile(tsrc)):
                log(ts=ts, op="sedr", ok=False, src=src, dst=dst); continue
            r = subprocess.run(["sed"] + args[:-1] + [tsrc], capture_output=True, text=True)
            if r.returncode == 0:
                open(tdst, "w").write(r.stdout)
            log(ts=ts, op="sedr", ok=r.returncode == 0, src=tsrc[len(TREE):], dst=tdst[len(TREE):], err=r.stderr[:200])
        elif m.group("cp") or m.group("mv"):
            op = "cp" if m.group("cp") else "mv"
            args = words_from(cmd, m.end())
            if args is None: continue
            args = [subst(a, env) for a in args]
            args = [a for a in args if not a.startswith("-")]
            if len(args) < 2: continue
            srcs, dst = [resolve(a, cwd, env) for a in args[:-1]], resolve(args[-1], cwd, env)
            tdst = to_tree(dst)
            if not tdst: continue
            for s_ in srcs:
                ts_ = to_tree(s_)
                if not ts_ or not os.path.isfile(ts_):
                    log(ts=ts, op=op, ok=False, src=s_, dst=dst, why="src missing in tree"); continue
                d = os.path.join(tdst, os.path.basename(ts_)) if os.path.isdir(tdst) else tdst
                os.makedirs(os.path.dirname(d), exist_ok=True)
                (shutil.copy2 if op == "cp" else shutil.move)(ts_, d)
                log(ts=ts, op=op, ok=True, src=ts_[len(TREE):], dst=d[len(TREE):])
snapshot(_prev_ts or "9999")
LOG.close()

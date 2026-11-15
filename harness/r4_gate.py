"""R4: distractor-tolerant reduce as a gated switch. The baseline answer (prompt B, kept evidence) stands unless a second
reduce over the SAME kept evidence with the abstaining prompt A disagrees with it or abstains; then the answer from the
enlarged-evidence run (TARGET: R3 second opinion or R2 V3 rank prior, both already computed) replaces it.

usage: r4_gate.py --qa QA.json --target r3_full_S2_mini|san_full_r2v3 --equality strict|semantic --out ans/NAME.jsonl [--score]
env as the baseline (ITEMS_FILE items-san, BEST_GUESS=1, MAP_MODEL flash, ANSWER_MODEL mini). The map is replayed from cache
with the B system prompt; only the A-prompt reduce (and, for semantic, a flash SAME/DIFFERENT check) is new spend.
--score: print the gated net against the baseline from the EXISTING per-question judgments (subset selection stage only).
"""
import argparse, json, os, re, sys, concurrent.futures as cf
SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP)
import llm, mf_answer_v2 as M
SYSTEM_A = ("You answer questions about the user's own personal memories (photos, videos, emails) using ONLY the memory "
            "items given. Each item is labelled by its real ID and timestamp. If the items are insufficient, say 'Unknown'.")
EQ_MODEL = os.environ.get("EQ_MODEL", "deepseek/deepseek-v4.1-flash")
ap = argparse.ArgumentParser(); ap.add_argument("--qa", required=True); ap.add_argument("--target", required=True)
ap.add_argument("--equality", default="strict"); ap.add_argument("--out", required=True); ap.add_argument("--score", action="store_true")
ap.add_argument("--workers", type=int, default=6)
a = ap.parse_args()
qa = json.load(open(a.qa)); rank = json.load(open(SP + "/runs/san_fused.rank.json"))
base = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(SP + "/ans/san_full.jsonl")}
targ = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(SP + f"/ans/{a.target}.jsonl")}
ABST = re.compile(r"\bunknown\b|not (?:stated|mentioned|specified|provided|found)|insufficient|cannot be determined|no (?:item|memory|information)", re.I)
IDS = re.compile(r"\b(?:email\d{12}|\d{8}_\d{6})\b")
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip().rstrip(". ")

def answer_A(q):
    pool = [i for i in rank[q["question"]][:25] if i in M.ITEMS]
    kept = []
    for s in range(0, len(pool), 10):
        kept += [i for i, _ in M.map_batch(q["question"], pool[s:s + 10])]   # cached: B system prompt
    ids = sorted(dict.fromkeys(kept), key=M.ts) or sorted(pool, key=M.ts)
    msgs = M.reduce_prompt(q["question"], "\n\n".join(M.item_text(i) for i in ids), q["qtype"])
    msgs[0]["content"] = SYSTEM_A
    return M.final(M.chat(msgs))

def same(q, x, y):
    if q["qtype"] == "number":
        return norm(x) == norm(y)
    if q["qtype"] == "list_recall":
        return set(IDS.findall(x)) == set(IDS.findall(y))
    if norm(x) == norm(y) or a.equality == "strict":
        return norm(x) == norm(y)
    out = llm.chat([{"role": "user", "content": f"Question: {q['question']}\nAnswer 1: {x}\nAnswer 2: {y}\n\nDo the two answers give the same answer to the question (same fact, even if worded differently)? Reply with exactly SAME or DIFFERENT."}],
                   EQ_MODEL, max_tokens=5, tag="eq")
    return out.strip().upper().startswith("SAME")

def decide(q):
    i = q["id"]
    if base[i] == targ[i]:
        return i, False, None
    A = answer_A(q)
    fire = bool(ABST.search(A)) or not same(q, A, base[i])
    return i, fire, A

rows = {}
with cf.ThreadPoolExecutor(a.workers) as ex:
    for i, fire, A in ex.map(decide, qa):
        rows[i] = (fire, A)
with open(a.out, "w") as f:
    for q in qa:
        fire, A = rows[q["id"]]
        f.write(json.dumps({"id": q["id"], "answer": targ[q["id"]] if fire else base[q["id"]]}) + "\n")
json.dump({i: {"fire": fi, "A": A} for i, (fi, A) in rows.items()}, open(a.out.replace(".jsonl", ".gate.json"), "w"), indent=0)
ch = [q for q in qa if base[q["id"]] != targ[q["id"]]]; fired = [q for q in ch if rows[q["id"]][0]]
print(f"{a.target} {a.equality}: {len(qa)} q, target differs on {len(ch)}, gate fires on {len(fired)} (abstain {sum(1 for q in fired if ABST.search(rows[q['id']][1] or ''))}); {llm.cost_line()}", flush=True)
if a.score:
    b = {e["id"]: float(e["accuracy"]) for e in json.load(open(SP + "/ans/san_full.eval/atm_openai_gpt-5-mini.json"))}
    t = {e["id"]: float(e["accuracy"]) for e in json.load(open(SP + f"/ans/{a.target}.eval/atm_openai_gpt-5-mini.json"))}
    net = sum(t[q["id"]] - b[q["id"]] for q in fired); allsw = sum(t[q["id"]] - b[q["id"]] for q in ch)
    wrong = [q for q in ch if b[q["id"]] < 1]; right = [q for q in ch if b[q["id"]] == 1]
    rho = sum(1 for q in wrong if rows[q["id"]][0]) / max(1, len(wrong)); phi = sum(1 for q in right if rows[q["id"]][0]) / max(1, len(right))
    print(f"  existing judgments: gated net {net:+.1f} correct on {len(qa)} q (switch-all {allsw:+.1f}); baseline-wrong caught {rho:.2f} ({len(wrong)} q), baseline-right flagged {phi:.2f} ({len(right)} q)", flush=True)

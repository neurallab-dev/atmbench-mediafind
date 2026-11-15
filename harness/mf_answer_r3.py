"""R3: a second opinion on the top-K items the (flash) relevance map rejected, by a stronger model.

Wraps mf_answer_v2: map and reduce prompts unchanged; after the map, the rejected items among the fused top-K go to
SECOND_MODEL with the same relevance instruction, and whatever it keeps joins the reduce input. Nothing is removed.
env: SECOND_MODEL (required), SECOND_K (default 5), SECOND_CTX=1 (variant S2: the second pass also sees the one-line
facts the first pass kept, so it judges what is additionally needed), plus mf_answer_v2's own env.
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mf_answer_v2 as M

K = int(os.environ.get("SECOND_K", "5"))
SECOND_MODEL = os.environ["SECOND_MODEL"]
SECOND_CTX = os.environ.get("SECOND_CTX") == "1"


def second_opinion(q, rejected, kept):
    block = "\n\n".join(M.item_text(i) for i in rejected)
    ctx = ""
    if SECOND_CTX and kept:
        ctx = "Items already selected as relevant by a first pass (for context; do not repeat them):\n" + \
              "\n".join(f"{i} | {f}" for i, f in kept) + "\n\n"
    out = M.chat([{"role": "system", "content": M.SYSTEM},
                  {"role": "user", "content": f"Question: {q}\n\n{ctx}{M.MAP_INSTR}\n\nMemory items:\n{block}"}],
                 max_tokens=1200, model=SECOND_MODEL)
    valid = set(rejected)
    keep = []
    for line in out.splitlines():
        m = re.match(r"\s*[-*]?\s*(\S+?)\s*\|\s*(.*)", line)
        if m and m.group(1) in valid:
            keep.append((m.group(1), m.group(2).strip()))
    return keep


def decide(q, pool, batch=10):
    """(first-pass kept, rejected top-K, second-pass kept) for one question; all three lists of (id, fact)."""
    kept = []
    for s in range(0, len(pool), batch):
        kept += M.map_batch(q["question"], pool[s:s + batch])
    kept_ids = {i for i, _ in kept}
    rejected = [i for i in pool[:K] if i not in kept_ids]
    second = second_opinion(q["question"], rejected, kept) if rejected else []
    return kept, rejected, second


def answer(q, pool, mode, batch):
    qtype = q.get("qtype", "open_end")
    pool = [i for i in pool if i in M.ITEMS]
    if mode == "single":
        return M.answer(q, pool, mode, batch)
    kept, rejected, second = decide(q, pool, batch)
    ids = sorted(dict.fromkeys(i for i, _ in kept + second), key=M.ts)
    if not ids:
        ids = sorted(pool, key=M.ts)  # both passes kept nothing: whole pool, as before
    if not ids:
        return "Unknown"
    msgs = M.reduce_prompt(q["question"], "\n\n".join(M.item_text(i) for i in ids), qtype)
    try:
        ans = M.final(M.chat(msgs, max_tokens=8000 if M.REDUCE_REASONING else 1500, reasoning=M.REDUCE_REASONING))
    except Exception as e:
        print("reduce fallback", q.get("id"), repr(e)[:120], file=sys.stderr, flush=True)
        ans = M.final(M.chat(msgs, max_tokens=1500, model=M.FALLBACK_MODEL))
    if qtype == "list_recall":
        ans = M.clean_ids(ans, pool)
    return ans


M.answer = answer

if __name__ == "__main__":
    print(f"second opinion by {SECOND_MODEL} on rejected top-{K} (ctx={SECOND_CTX})", file=sys.stderr, flush=True)
    M.main()
    print(M.llm.cost_line(), file=sys.stderr, flush=True)

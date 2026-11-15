"""R2 rank prior: identical to mf_answer_v2.py except that the reduce step also receives the fused top-K items
(env RANK_PRIOR_K, default 5; RANK_PRIOR_K_LIST for list_recall questions, default = RANK_PRIOR_K) in addition to
whatever the relevance map kept. Prompts, models, map batches and the kept-nothing fallback are unchanged.
usage: same flags as mf_answer_v2.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mf_answer_v2 as M

K = int(os.environ.get("RANK_PRIOR_K", "5"))
K_LIST = int(os.environ.get("RANK_PRIOR_K_LIST", str(K)))


def answer(q, pool, mode, batch):
    qtype = q.get("qtype", "open_end")
    pool = [i for i in pool if i in M.ITEMS]
    if mode == "single":
        return M.answer(q, pool, mode, batch)
    kept = []
    for s in range(0, len(pool), batch):
        kept += M.map_batch(q["question"], pool[s:s + batch])
    ids = [i for i, _ in kept]
    if ids:
        ids = ids + pool[: (K_LIST if qtype == "list_recall" else K)]  # the rank prior
    ids = sorted(dict.fromkeys(ids), key=M.ts)
    if not ids:
        ids = sorted(pool, key=M.ts)  # map kept nothing: whole pool, as before
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
    if qtype == "number":
        ans = M.format_number_answer(ans)
    return ans


M.answer = answer
if __name__ == "__main__":
    print(f"rank prior K={K} K_LIST={K_LIST}", file=sys.stderr, flush=True)
    M.main()
    print(M.llm.cost_line(), file=sys.stderr, flush=True)

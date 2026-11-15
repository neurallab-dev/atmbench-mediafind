"""$0 sizing of the relevance-map stage: rebuild the flash map decisions of the submitted run (san_full: BEST_GUESS=1,
MAP_MODEL flash, pool = san_fused top-25, batch 10) from the on-disk LLM cache by recomputing the request keys. No API calls.
Reports, per question, which gold ids were in the pool, which the map kept, how many distractors it kept, joined with the
gpt-5-mini judgments of ans/san_full.eval."""
import os, sys, json, hashlib, re, collections
SP = os.path.dirname(os.path.abspath(__file__)); os.chdir(SP); sys.path.insert(0, SP)
os.environ["ITEMS_FILE"] = "atm/items-san.json"; os.environ["BEST_GUESS"] = "1"; os.environ["MAP_MODEL"] = "deepseek/deepseek-v4.1-flash"
os.environ["ANSWER_MODEL"] = "openai/gpt-5.4-mini"
import llm, mf_answer_v2 as M
llm._cache_file(M.MAP_MODEL)
qa = json.load(open("atm/atm-bench.json")); rank = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "runs/san_fused.rank.json"))
def key_for(messages, model, max_tokens):
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    r = llm.reasoning_for(model)
    if r is not None: body["reasoning"] = r
    if model.startswith("deepseek/"): body["provider"] = {"ignore": ["GMICloud"]}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
rows = []; missing = 0
for q in qa:
    pool = [i for i in rank.get(q["question"], [])[:25] if i in M.ITEMS]
    kept = []; miss_batches = 0
    for s in range(0, len(pool), 10):
        ids = pool[s:s+10]; block = "\n\n".join(M.item_text(i) for i in ids)
        msgs = [{"role": "system", "content": M.SYSTEM}, {"role": "user", "content": f"Question: {q['question']}\n\n{M.MAP_INSTR}\n\nMemory items:\n{block}"}]
        k = key_for(msgs, M.MAP_MODEL, 1200); out = llm._cache.get(k)
        if out is None: miss_batches += 1; continue
        valid = set(ids)
        for line in out.splitlines():
            m = re.match(r"\s*[-*]?\s*(\S+?)\s*\|\s*(.*)", line)
            if m and m.group(1) in valid: kept.append(m.group(1))
    if miss_batches: missing += 1
    gold = set(q["evidence_ids"]); inpool = gold & set(pool); keptset = set(kept)
    rows.append({"id": q["id"], "qtype": q.get("qtype"), "n_gold": len(gold), "gold_in_pool": len(inpool), "gold_kept": len(inpool & keptset),
                 "kept": len(keptset), "distractors_kept": len(keptset - gold), "fallback_whole_pool": len(keptset) == 0, "miss_batches": miss_batches})
print("questions", len(rows), "with uncached map batches", missing)
json.dump(rows, open("runs/map_loss.json", "w"))
c = collections.Counter()
for r in rows:
    b = ("gold not all in pool" if r["gold_in_pool"] < r["n_gold"] else "all gold kept" if r["gold_kept"] == r["gold_in_pool"] else "map dropped gold")
    c[b] += 1
print(c)
print("kept per q: mean %.1f  distractors mean %.1f  whole-pool fallbacks %d" % (sum(r["kept"] for r in rows)/len(rows), sum(r["distractors_kept"] for r in rows)/len(rows), sum(r["fallback_whole_pool"] for r in rows)))

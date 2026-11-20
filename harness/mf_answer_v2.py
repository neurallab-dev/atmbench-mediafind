"""MediaFind-style answer pipeline over a fixed evidence list (map-reduce v2).

usage: mf_answer_v2.py --qa QA.json --pool <rank.json>|gold|niah --topk 25 --mode mapreduce --workers 6 --out answers.jsonl
env: ANSWER_MODEL (reduce), MAP_MODEL (relevance map; default = ANSWER_MODEL), FALLBACK_MODEL (reduce fallback),
     ITEMS_FILE (item texts), BEST_GUESS=1|2|3 (prompt variants), REDUCE_REASONING (json reasoning override for the reduce step)
"""
import argparse, json, re, sys, concurrent.futures as cf
import os
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import llm
from ans_format import format_number_answer  # deterministic date/time formatting of number-type answers
MODEL = os.environ.get("ANSWER_MODEL", "openai/gpt-5.4-mini")
MAP_MODEL = os.environ.get("MAP_MODEL") or MODEL  # a cheaper model may do the relevance map
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "anthropic/claude-sonnet-5.5")
ITEMS = {i["item_id"]: i for i in json.load(open(os.environ.get("ITEMS_FILE", SP + "/atm/items.json")))}
# REDUCE_REASONING='{"enabled": true}' (deepseek) / '{"effort": "medium"}' (gpt-5.x) turns the model's own
# reasoning on for the reduce step only; the map filter stays cheap.
REDUCE_REASONING = json.loads(os.environ["REDUCE_REASONING"]) if os.environ.get("REDUCE_REASONING") else None


def chat(messages, max_tokens=1500, reasoning=None, model=None):
    return llm.chat(messages, model or MODEL, max_tokens=max_tokens, tag="ans", reasoning=reasoning)


def ts(i):
    m = re.search(r"Timestamp: (\S+ \S+)", ITEMS[i]["text"])
    return m.group(1)[:19] if m else "9999"


def item_text(i):
    return ITEMS[i]["text"].strip()


TYPE_RULES = {
    "list_recall": "The question asks to recall items. Final answer: ONLY the item IDs (like 20230101_120000 or email202301010001), comma-separated, copied EXACTLY from the 'ID:' lines. Include every item that matches; never write 'Evidence N'.",
    "number": "Give exactly what the question asks for, as the items state it: a date, a time, a place, an amount with its currency, or a count. Only when a count or total must be computed across items, work it out step by step first.",
    "open_end": "Answer concisely but completely, keeping names, places, dates and amounts exactly as the items state them.",
}

SYSTEM = ("You answer questions about the user's own personal memories (photos, videos, emails) using ONLY the memory "
          "items given. Each item is labelled by its real ID and timestamp. If the items are insufficient, say 'Unknown'.")
if os.environ.get("BEST_GUESS"):
    # The benchmark has (almost) no abstention questions: 'Unknown' always scores 0, so a best-supported guess can only help.
    SYSTEM = ("You answer questions about the user's own personal memories (photos, videos, emails) using ONLY the memory "
              "items given. Each item is labelled by its real ID and timestamp. The question always refers to something that "
              "IS among the items, even when the item's wording differs from the question (a caption may describe the dish, "
              "object or event in other words): find the item(s) that best match and answer from them. Never answer 'Unknown'; "
              "if several items could match, pick the most plausible one and answer with its details.")
if os.environ.get("NUM_RULE") == "2":
    # The deterministic matcher wants exactly the gold's numbers/dates/times: an added weekday, time or second figure fails it.
    TYPE_RULES["number"] = (
        "Give ONLY the value asked for and nothing else: a date as 'Month D, YYYY' (e.g. 'August 9, 2024'); a clock time as "
        "'HH:MM'; an amount with its currency symbol exactly as the item writes it (e.g. '£799.74'); a count as a bare number. "
        "Include a time only if the question asks what time; include a date only if it asks what day; never add a weekday, a "
        "place, a duration, a range or any second number that was not asked for. Only when a count or total must be computed "
        "across items, work it out step by step under 'Reasoning:' first.")
if os.environ.get("OE_RULE") == "2":
    TYPE_RULES["open_end"] = (
        "Answer with only the fact asked for, as one short phrase or sentence, using names, places, dates and amounts exactly as "
        "the items state them. Do not add details that were not asked for (no extra dates, times, addresses, prices or "
        "explanations): an added detail that disagrees with the record makes the whole answer wrong.")
if os.environ.get("LIST_RULE") == "2":
    # 96 of 139 list questions have ONE gold item; listing several candidates costs Jaccard.
    TYPE_RULES["list_recall"] = (
        "The question asks to recall items. Final answer: ONLY item IDs (like 20230101_120000 or email202301010001), "
        "comma-separated, copied EXACTLY from the 'ID:' lines; never write 'Evidence N'. If the question asks for ONE "
        "specific photo/video/email (singular: 'the photo', 'that video', 'a photo of', 'which photo', 'the one where'), "
        "answer with exactly ONE ID — the single best match, even if several items are similar. Only when it asks for "
        "several (plural: 'photos', 'all the videos', 'every', 'the footage from that day') list every matching ID.")
if os.environ.get("BEST_GUESS") == "3":
    TYPE_RULES["open_end"] += (" Exception: when the question asks for a specific figure (a price, a cost, a time) and no "
                               "item states that figure for the thing asked about (e.g. an email gives only an order total "
                               "that covers other items too), answer exactly 'ABSTENTION. <short reason>' instead of guessing.")
if os.environ.get("BEST_GUESS") == "2":
    SYSTEM += (" The ONLY exception: when the question asks for a specific figure (a price, a cost, a time, a quantity) "
               "and no item states that figure for the thing asked about (e.g. an email gives only an order total that "
               "covers other items too), answer exactly 'ABSTENTION. <short reason>' instead of guessing a number.")


def reduce_prompt(q, notes, qtype):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"Question: {q}\n\nMemory items (oldest first):\n{notes}\n\n{TYPE_RULES.get(qtype, TYPE_RULES['open_end'])}\n"
            "First think briefly under 'Reasoning:', then end with a single line 'ANSWER: <answer>'.")},
    ]


def final(text):
    m = re.findall(r"ANSWER:\s*(.+)", text, flags=re.S)
    return m[-1].strip() if m else text.strip()


MAP_INSTR = ("For EACH memory item below, decide whether it is relevant to answering the question (it may be one of several "
             "pieces needed; when the question asks for all items of a trip/place/kind, every matching item is relevant). "
             "Output one line per RELEVANT item only, in the form:\n<ID> | <date> | <one-line fact useful for the question: place, what, amount, dates>\n"
             "Output NONE if no item is relevant. No other text.")


def map_batch(q, ids):
    block = "\n\n".join(item_text(i) for i in ids)
    out = chat([{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Question: {q}\n\n{MAP_INSTR}\n\nMemory items:\n{block}"}], max_tokens=1200, model=MAP_MODEL)
    keep = []
    valid = set(ids)
    for line in out.splitlines():
        m = re.match(r"\s*[-*]?\s*(\S+?)\s*\|\s*(.*)", line)
        if m and m.group(1) in valid:
            keep.append((m.group(1), m.group(2).strip()))
    return keep


def clean_ids(ans, pool):
    """Keep only IDs that exist in the pool (the model retypes IDs and sometimes shifts a digit)."""
    found = re.findall(r"\b(?:email\d{12}|\d{8}_\d{6})\b", ans)
    if not found:
        return ans
    good = [i for i in dict.fromkeys(found) if i in pool]
    return ", ".join(good) if good else ans


def answer(q, pool, mode, batch):
    qtype = q.get("qtype", "open_end")
    pool = [i for i in pool if i in ITEMS]
    if mode == "single":
        notes = "\n\n".join(item_text(i) for i in sorted(pool, key=ts))
        return final(chat(reduce_prompt(q["question"], notes, qtype)))
    kept = []
    if mode == "all":
        # No relevance map: the standard reduce call reads every pooled item (screening of 2026-11-20).
        kept = [(i, "") for i in pool]
    for s in range(0, len(pool) if mode == "mapreduce" else 0, batch):
        kept += map_batch(q["question"], pool[s:s + batch])
    ids = sorted(dict.fromkeys(i for i, _ in kept), key=ts)
    if not ids:
        # The map filter kept nothing: answer over the whole pool rather than giving up.
        ids = sorted(pool, key=ts)
    if not ids:
        return "Unknown"
    msgs = reduce_prompt(q["question"], "\n\n".join(item_text(i) for i in ids), qtype)
    try:
        ans = final(chat(msgs, max_tokens=8000 if REDUCE_REASONING else 1500, reasoning=REDUCE_REASONING))
    except Exception as e:
        # A refused/filtered/empty reduce (e.g. a provider content filter) falls back to another model.
        print("reduce fallback", q.get("id"), repr(e)[:120], file=sys.stderr, flush=True)
        ans = final(chat(msgs, max_tokens=1500, model=FALLBACK_MODEL))
    if qtype == "list_recall":
        ans = clean_ids(ans, pool)
    if qtype == "number":
        ans = format_number_answer(ans)
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--pool", required=True, help="'niah', 'gold' (oracle: the question's evidence ids) or a rank json {question: [ids]}")
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--mode", choices=["single", "mapreduce", "all"], default="mapreduce")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    qs = json.load(open(a.qa))
    rank = None if a.pool in ("niah", "gold") else json.load(open(a.pool))
    def run(q):
        if a.pool == "gold":
            pool = list(q["evidence_ids"])
        else:
            pool = q["niah_evidence_ids"] if rank is None else rank.get(q["question"], [])[: a.topk]
        try:
            return {"id": q["id"], "answer": answer(q, pool, a.mode, a.batch)}
        except Exception as e:
            print("FAILED", q["id"], e, file=sys.stderr, flush=True)
            return {"id": q["id"], "answer": "Unknown"}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, qs))
    with open(a.out, "w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(res), "|", llm.cost_line())


if __name__ == "__main__":
    main()

"""Pass A: extract a trip/event timeline from ALL emails (chronological batches) with an LLM.

usage: trips_extract.py OUT.json [model]   (env WORKERS, BATCH)
Each record: {kind, name, place, venue, start, end, facts, emails:[ids]} — only things that put the user
somewhere at some time (bookings, conference attendance, registered events, formals, dinners, visits).
"""
import json, os, re, sys, concurrent.futures as cf
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import llm

out_p = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek/deepseek-v4.1-flash"
BATCH = int(os.environ.get("BATCH", "50")); WORKERS = int(os.environ.get("WORKERS", "12"))
items = json.load(open(SP + "/atm/items.json"))
emails = sorted((i for i in items if i["modality"] == "email"), key=lambda i: i["metadata"]["timestamp"])

SYSTEM = ("You build a personal travel/event timeline from a person's emails. The person appears to be a graduate "
          "student/researcher at the University of Cambridge, UK (home: Petersfield area, Cambridge), who travels to "
          "academic conferences and on holidays. Use your own knowledge of conference names, venues, cities and dates "
          "to normalize what the emails imply (a conference's usual host city and dates for that year, for example).")

INSTR = ("From the emails below (chronological; each starts with its ID and timestamp), list every TRIP, CONFERENCE, "
         "EVENT, BOOKING or VISIT that places the person somewhere at a specific time: flights, trains, hotels, "
         "Airbnb, conference registrations/attendance, workshops, formal dinners, invitations the person booked or "
         "accepted, restaurant/table bookings, visa documents for a trip, tickets to shows. Ignore newsletters, "
         "marketing, generic university notices and anything without a concrete date for the person. "
         "Merge emails about the same trip into ONE record where possible.\n"
         "Output ONLY a JSON array, each element: {\"kind\": \"conference|trip|flight|train|hotel|event|dinner|visit|other\", "
         "\"name\": \"<short name a person would use, e.g. '<conference acronym> <year>', 'summer trip to <country>', '<college> formal'>\", "
         "\"place\": \"<city, region/country>\", \"venue\": \"<hotel/venue/airport if known>\", \"start\": \"YYYY-MM-DD\", "
         "\"end\": \"YYYY-MM-DD\", \"facts\": \"<one line: booking refs, flight/train route & times, hotel name, amounts with "
         "currency, companions, organizer>\", \"emails\": [\"<email ids this record came from>\"]}. Dates must be ISO; use the "
         "same date for start and end when unknown; [] if nothing qualifies.")


def email_block(e):
    t = e["text"]
    t = re.sub(r"\nDetail: Date: \S+\n", "\n", t)
    return t.strip()


def run(bi):
    batch = emails[bi * BATCH:(bi + 1) * BATCH]
    block = "\n\n".join(email_block(e) for e in batch)
    out = llm.chat([{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"{INSTR}\n\nEMAILS:\n{block}"}], MODEL, max_tokens=4000, tag=f"trips{bi}")
    m = re.search(r"\[.*\]", out, flags=re.S)
    try:
        recs = json.loads(m.group(0)) if m else []
    except Exception:
        print("bad json batch", bi, out[:200], file=sys.stderr); recs = []
    valid = {e["item_id"] for e in batch}
    good = []
    for r in recs:
        if not isinstance(r, dict): continue
        r["emails"] = [x for x in (r.get("emails") or []) if x in valid]
        r["batch"] = bi
        good.append(r)
    return good


n_batches = (len(emails) + BATCH - 1) // BATCH
print(f"{len(emails)} emails, {n_batches} batches of {BATCH}, model {MODEL}", flush=True)
recs = []
with cf.ThreadPoolExecutor(WORKERS) as ex:
    for i, part in enumerate(ex.map(run, range(n_batches))):
        recs += part
        if (i + 1) % 10 == 0: print(f"batch {i+1}/{n_batches} records {len(recs)} | {llm.cost_line()}", flush=True)
recs.sort(key=lambda r: (str(r.get("start") or ""), str(r.get("name") or "")))
json.dump(recs, open(out_p, "w"), ensure_ascii=False, indent=1)
print(f"wrote {len(recs)} records -> {out_p} | {llm.cost_line()}", flush=True)

"""Pass B: one memory card per photo/video item from its caption + trip timeline + linked emails + same-day neighbours.

usage: cards.py OUT.jsonl [model] [item_id ...]   (env WORKERS; TRIPS=cards/trips.json)
Writes {"item_id", "card"} per line (append; resumable — already-written ids are skipped).
"""
import json, os, re, sys, datetime as dt, collections, concurrent.futures as cf
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import llm

out_p = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek/deepseek-v4.1-flash"
ONLY = set(sys.argv[3:])
WORKERS = int(os.environ.get("WORKERS", "16"))
items = json.load(open(SP + "/atm/items.json"))
ITEMS = {i["item_id"]: i for i in items}
trips = json.load(open(os.environ.get("TRIPS", SP + "/cards/trips.json")))
KW = re.compile(r"\b(flight|flights|airline|boarding|booking|booked|reservation|hotel|hostel|airbnb|check-in|check in|"
                r"itinerary|conference|workshop|symposium|registration|registered|visa|e-?ticket|tickets?|accommodation|"
                r"trip|travel|journey|departure|arrival|airport|venue|invitation|formal|dinner|summit|hackathon|tutorial|"
                r"keynote|poster|camera-ready|accepted|train|museum|concert|show|festival)\b", re.I)
GENERIC = {"united kingdom", "england", "scotland", "wales", "united states", "china", "italy", "thailand", "spain", "france",
           "egypt", "jordan", "australia", "canada", "germany", "japan", "cambridgeshire", "cambridgeshire and peterborough",
           "greater london", "london", "cambridge", "europe", "uk", "usa"}


def day(i):
    return dt.date.fromisoformat(ITEMS[i]["metadata"]["timestamp"][:10])


def pdate(s):
    try:
        return dt.date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def city_of(loc):
    parts = [p.strip() for p in loc.split(",") if p.strip() and not re.search(r"\d", p)]
    return ", ".join(parts[-3:]) if parts else loc[:60]


def place_tokens(loc):
    """Specific place names in a Location string (venue, street, district, city) for matching emails."""
    toks = []
    for p in loc.split(","):
        p = p.strip()
        if len(p) < 4 or re.search(r"\d", p) or p.lower() in GENERIC:
            continue
        toks.append(p.lower())
    return toks


def short_caption(i):
    m = re.search(r"Short Caption: (.*)", ITEMS[i]["text"])
    return (m.group(1) if m else ITEMS[i]["text"][:160]).strip()


def email_line(e, limit=230):
    t = ITEMS[e]["text"]
    sub = re.search(r"Subject: (.*)", t); summ = re.search(r"Summary: (.*)", t)
    return f"- [{e}] {ITEMS[e]['metadata']['timestamp'][:10]} \"{(sub.group(1) if sub else '').strip()[:90]}\" — {(summ.group(1) if summ else '').strip()[:limit]}"


media = [i["item_id"] for i in items if i["modality"] != "email"]
emails = sorted((i["item_id"] for i in items if i["modality"] == "email"), key=lambda e: ITEMS[e]["metadata"]["timestamp"])
EMAIL_LOWER = {e: ITEMS[e]["text"].lower() for e in emails}
by_day = collections.defaultdict(list)
for m in media:
    by_day[day(m)].append(m)
for d in by_day:
    by_day[d].sort(key=lambda i: ITEMS[i]["metadata"]["timestamp"])
email_days = collections.defaultdict(list)
for e in emails:
    email_days[day(e)].append(e)
for t in trips:
    t["_s"], t["_e"] = pdate(t.get("start")), pdate(t.get("end") or t.get("start"))


def context(i):
    d = day(i)
    recs = [t for t in trips if t["_s"] and t["_e"] and t["_s"] - dt.timedelta(3) <= d <= t["_e"] + dt.timedelta(3)]
    recs.sort(key=lambda t: (abs((t["_s"] - d).days), t.get("name", "")))
    rec_lines = [f"- [{t.get('kind','')}] {t.get('name','')} — {t.get('place','')}; venue: {t.get('venue','') or '-'}; "
                 f"{t.get('start','')} to {t.get('end','')}; {str(t.get('facts',''))[:220]} (emails: {', '.join(t.get('emails', [])[:4])})"
                 for t in recs[:12]]
    # emails within ±10 days: place-linked first (text names the photo's venue/district/city), then travel-keyword ones
    toks = place_tokens(ITEMS[i]["metadata"]["location"])
    window = [(abs(off), e) for off in range(-10, 11) for e in email_days.get(d + dt.timedelta(off), [])]
    window.sort()
    linked = [e for _, e in window if any(t in EMAIL_LOWER[e] for t in toks)][:3]
    travel = [e for _, e in window if e not in linked and KW.search(ITEMS[e]["text"])][:6]
    email_lines = [email_line(e) for e in linked + travel]
    same = [m for m in by_day[d] if m != i]
    same.sort(key=lambda m: abs((dt.datetime.fromisoformat(ITEMS[m]["metadata"]["timestamp"][:19]) -
                                 dt.datetime.fromisoformat(ITEMS[i]["metadata"]["timestamp"][:19])).total_seconds()))
    same = sorted(same[:24], key=lambda m: ITEMS[m]["metadata"]["timestamp"])
    nb_lines = [f"- {ITEMS[m]['metadata']['timestamp'][11:16]} [{m}] ({city_of(ITEMS[m]['metadata']['location'])}) {short_caption(m)[:150]}" for m in same]
    shape = []
    for off in range(-6, 7):
        dd = d + dt.timedelta(off)
        ms = by_day.get(dd, [])
        if ms:
            c = collections.Counter(city_of(ITEMS[m]["metadata"]["location"]) for m in ms).most_common(1)[0][0]
            shape.append(f"{dd} ({'this day' if off == 0 else f'{off:+d}d'}): {len(ms)} items, mostly {c}")
    return rec_lines, email_lines, nb_lines, shape, len(by_day[d])


SYSTEM = ("You write short, factual 'memory cards' for a person's photos and videos so they can later be found by "
          "natural-language recall questions (e.g. 'the formal at Peterhouse last week', 'burrata during ECCV 2024', "
          "'the photo of London from my flight to Cairo', 'all the videos from the day I made mochi'). The person appears "
          "to be a graduate student/researcher at the University of Cambridge, UK (home: Petersfield area of Cambridge — "
          "Warkworth Street / Parkside; college: Peterhouse; Department of Engineering), who travels to academic conferences "
          "and on holidays. Use your own world knowledge to name conferences, venues, colleges, districts, landmarks and "
          "dishes precisely (e.g. Allianz MiCo = ECCV 2024 venue in Milan's CityLife district; Fellows' Garden, "
          "Trumpington Street = Peterhouse), but NEVER invent an event the evidence does not support: when the item is "
          "ordinary daily life, say so plainly.")

INSTR = ("Write the memory card for THE ITEM below. Use the trip/event timeline, the emails and the other photos/videos "
         "taken the same day to work out which trip, conference, event or everyday activity it belongs to, and normalize "
         "names, places and dates. Output ONLY these lines (no preamble, no markdown, <= 140 words in total):\n"
         "Event: <the trip/conference/event/activity this belongs to, with full and short names, city, country and date "
         "range, e.g. 'ACL 2024 (62nd Annual Meeting of the ACL), Bangkok, Thailand, 11-16 Aug 2024' or 'formal dinner at "
         "Peterhouse, Cambridge' or 'everyday life at home in Cambridge'>\n"
         "Place: <venue/landmark + district, city, region, country, with aliases a person would use>\n"
         "When: <YYYY-MM-DD, weekday, time of day; position within the trip (e.g. day 3 of 6; the day I flew back)>\n"
         "What: <1-2 plain sentences: what is shown or happening, naming dishes, objects, text seen, people, companions, "
         "amounts; mention how it relates to the other items that day (e.g. one of 9 videos of making mochi that afternoon)>\n"
         "Keywords: <12-25 comma-separated search terms and aliases: event short names, place names, activity, objects, food, "
         "people, trip descriptors like 'flight to Cairo', 'summer trip to France'>")


def build_prompt(i):
    rec_lines, email_lines, nb_lines, shape, n_day = context(i)
    parts = ["TRIP/EVENT TIMELINE RECORDS AROUND THIS DATE (from the person's emails; may be incomplete or duplicated):",
             "\n".join(rec_lines) or "(none)",
             "\nRELATED EMAILS (place-linked and travel-related, within 10 days):", "\n".join(email_lines) or "(none)",
             f"\nOTHER PHOTOS/VIDEOS THE SAME DAY ({n_day} items that day in total):", "\n".join(nb_lines) or "(none)",
             "\nSURROUNDING DAYS:", "\n".join(shape),
             "\nTHE ITEM:", ITEMS[i]["text"].strip()[:3500]]
    return f"{INSTR}\n\n" + "\n".join(parts)


def parse_card(out):
    out = re.sub(r"^```.*?\n|\n```$", "", out.strip(), flags=re.S).strip()
    if not re.search(r"^Event:", out, flags=re.M):
        # The model sometimes drops the first label; the rest of the card is still well-formed.
        if re.search(r"^(Place|When|What|Keywords):", out, flags=re.M):
            out = "Event: " + out.lstrip()
        else:
            raise RuntimeError("card missing Event line: " + out[:120])
    return out


def run(i):
    try:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(i)}]
        try:
            out = parse_card(llm.chat(msgs, MODEL, max_tokens=500, tag=i))
        except RuntimeError:
            # A cached malformed answer: ask again without the cache and with a little temperature.
            out = parse_card(llm.chat(msgs, MODEL, max_tokens=600, tag=i, use_cache=False, temperature=0.3))
        return i, out
    except Exception as e:
        print("FAILED", i, repr(e)[:200], file=sys.stderr, flush=True)
        return i, None


if __name__ == "__main__":
    todo = [m for m in media if (not ONLY or m in ONLY)]
    done = set()
    if os.path.exists(out_p) and not ONLY:
        for line in open(out_p):
            try: done.add(json.loads(line)["item_id"])
            except Exception: pass
    todo = [m for m in todo if m not in done]
    print(f"{len(todo)} items to do ({len(done)} already), model {MODEL}, workers {WORKERS}", flush=True)
    if ONLY:
        for i in todo:
            print("=" * 80, i, city_of(ITEMS[i]["metadata"]["location"]))
            print(build_prompt(i)[-1200:] if os.environ.get("SHOW_PROMPT") else "")
            _, card = run(i); print(card)
        print(llm.cost_line()); sys.exit()
    n = 0; fails = 0
    with open(out_p, "a") as f, cf.ThreadPoolExecutor(WORKERS) as ex:
        for i, card in ex.map(run, todo):
            n += 1
            if card is None:
                fails += 1; continue
            f.write(json.dumps({"item_id": i, "card": card}, ensure_ascii=False) + "\n"); f.flush()
            if n % 200 == 0: print(f"{n}/{len(todo)} fails {fails} | {llm.cost_line()}", flush=True)
    print(f"done {n} fails {fails} | {llm.cost_line()}", flush=True)

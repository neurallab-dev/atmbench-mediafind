"""Dense, question-agnostic visual descriptions of the raw photos/videos with a hosted VLM (OpenRouter).

usage: vlm_caption.py OUT.jsonl [model] [item_id ...]   (env WORKERS, MEDIA=~/atmbench/data/media)
Writes {"item_id", "desc"} per line; resumable. Videos: 4 evenly spaced frames in one request.
"""
import base64, io, json, os, re, subprocess, sys, tempfile, concurrent.futures as cf
from PIL import Image
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import llm

out_p = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek/deepseek-v4-flash-vision-exp"
ONLY = set(sys.argv[3:])
WORKERS = int(os.environ.get("WORKERS", "12"))
MEDIA = os.path.expanduser(os.environ.get("MEDIA", "~/atmbench/data/media"))
items = [i for i in json.load(open(SP + "/atm/items.json")) if i["modality"] != "email"]

PROMPT = ("Describe this photo for a personal photo-memory search index. Be exhaustive and literal, not poetic: list every "
          "object, animal, vehicle, food dish (by name if recognizable), brand, sign and ALL visible text verbatim (menus, "
          "screens, documents, receipts, tickets, posters, labels, number plates — including amounts, dates, times, names, "
          "addresses, IP addresses, codes), counts and colours, the weather and time of day, what people are doing and "
          "wearing, distinctive details someone would remember, "
          "and clues to the venue or event. Up to 200 words of plain prose or semicolon-separated phrases; no preamble; "
          "list each piece of text once — never repeat.")
PROMPT_VIDEO = ("These are 4 frames, in order, from one short video. " + PROMPT.replace("this photo", "the video"))


def img_part(img, max_side=1024, q=85):
    img = img.convert("RGB")
    w, h = img.size
    s = max_side / max(w, h)
    if s < 1:
        img = img.resize((int(w * s), int(h * s)))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=q)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}


def frames(path, n=4):
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                               capture_output=True, text=True).stdout.strip() or "0")
    out = []
    with tempfile.TemporaryDirectory() as td:
        for k in range(n):
            t = max(0.0, dur * (k + 0.5) / n)
            fp = os.path.join(td, f"f{k}.jpg")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1", "-vf", "scale=768:-2", fp],
                           capture_output=True)
            if os.path.exists(fp):
                out.append(img_part(Image.open(fp), max_side=768))
    return out


def dedupe(text, limit=1600):
    """Collapse degenerate repetition (a VLM looping over the same quoted strings) and cap the length."""
    segs = re.split(r"(?<=[;,.])\s+", text.strip())
    seen = {}; keep = []
    for seg in segs:
        k = seg.strip().lower()
        seen[k] = seen.get(k, 0) + 1
        if seen[k] <= 2:
            keep.append(seg.strip())
    out = " ".join(keep)
    return out[:limit]


def describe(it):
    i = it["item_id"]
    try:
        if it["modality"] == "image":
            path = os.path.join(MEDIA, "image", i + ".jpg")
            parts = [{"type": "text", "text": PROMPT}, img_part(Image.open(path))]
        else:
            path = os.path.join(MEDIA, "video", i + ".mp4")
            fr = frames(path)
            if not fr:
                raise RuntimeError("no frames")
            parts = [{"type": "text", "text": PROMPT_VIDEO}] + fr
        out = llm.chat([{"role": "user", "content": parts}], MODEL, max_tokens=600, tag="vlm:" + i)
        return i, dedupe(out)
    except Exception as e:
        print("FAILED", i, repr(e)[:200], file=sys.stderr, flush=True)
        return i, None


if __name__ == "__main__":
    todo = [it for it in items if (not ONLY or it["item_id"] in ONLY)]
    done = set()
    if os.path.exists(out_p) and not ONLY:
        for line in open(out_p):
            try: done.add(json.loads(line)["item_id"])
            except Exception: pass
    todo = [it for it in todo if it["item_id"] not in done]
    print(f"{len(todo)} items to do ({len(done)} already), model {MODEL}, workers {WORKERS}", flush=True)
    if ONLY:
        for it in todo:
            i, d = describe(it); print("=" * 70, i, it["modality"]); print(d)
        print(llm.cost_line()); sys.exit()
    n = fails = 0
    with open(out_p, "a") as f, cf.ThreadPoolExecutor(WORKERS) as ex:
        for i, d in ex.map(describe, todo):
            n += 1
            if d is None:
                fails += 1; continue
            f.write(json.dumps({"item_id": i, "desc": d}, ensure_ascii=False) + "\n"); f.flush()
            if n % 200 == 0: print(f"{n}/{len(todo)} fails {fails} | {llm.cost_line()}", flush=True)
    print(f"done {n} fails {fails} | {llm.cost_line()}", flush=True)

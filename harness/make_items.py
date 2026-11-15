"""Rebuild items.json (official MMRAG item texts) and the MediaFind tasks file from the dataset files.

usage: make_items.py   -> atm/items.json, atm/atm-bench.json (copy), atm/atm-bench.doc.jsonl
Text format = memqa/retrieve/utils.py format_media_text / format_email_text with every field on.
"""
import json, os, re, shutil
from datetime import datetime, timezone
SP = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(SP, "data")


def records(obj):
    if isinstance(obj, dict):
        for k in ("results", "items", "data"):
            if isinstance(obj.get(k), list):
                return obj[k]
        vals = list(obj.values())
        if vals and isinstance(vals[0], dict):
            return vals
        return [x for v in vals if isinstance(v, list) for x in v]
    return obj


def media_text(item, eid, etype):
    tags = item.get("tags", []) or []
    return "\n".join([
        f"ID: {eid}", f"Type: {etype}", f"Timestamp: {item.get('timestamp', '') or ''}",
        f"Location: {item.get('location_name', '') or ''}", f"Short Caption: {item.get('short_caption', '') or ''}",
        f"Caption: {item.get('caption', '') or ''}", f"OCR: {item.get('ocr_text', '') or ''}",
        "Tags: " + (", ".join(str(t) for t in tags) if isinstance(tags, list) else str(tags)),
    ])


def email_text(item):
    return "\n".join([f"ID: {item.get('id', '') or ''}", f"Timestamp: {item.get('timestamp', '') or ''}",
                      f"Summary: {item.get('short_summary', '') or ''}", f"Detail: {item.get('detail', '') or ''}"])


def epoch(value):
    when = datetime.fromisoformat(str(value))
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.timestamp()


items = []
for name, key, etype in (("processed_memory/image_batch_results.json", "image_path", "image"),
                         ("processed_memory/video_batch_results.json", "video_path", "video")):
    for it in records(json.load(open(os.path.join(D, name)))):
        raw = it.get(key) or it.get("path") or ""
        eid = os.path.splitext(os.path.basename(str(raw)))[0]
        items.append({"item_id": eid, "modality": etype, "text": media_text(it, eid, etype), "image_path": None, "video_path": None,
                      "metadata": {"source": etype, "batch_path": raw, "timestamp": it.get("timestamp", "") or "",
                                   "location": it.get("location_name", "") or ""}})
for it in records(json.load(open(os.path.join(D, "raw_memory/email/emails.json")))):
    detail = it.get("detail", "") or ""
    m = re.search(r"^Subject:\s*(.+)$", detail, flags=re.M)
    items.append({"item_id": it["id"], "modality": "email", "text": email_text(it), "image_path": None, "video_path": None,
                  "metadata": {"source": "email", "timestamp": it.get("timestamp", "") or "", "subject": m.group(1).strip() if m else "",
                               "sender": it.get("sender", "") or "", "location": ""}})
ids = [i["item_id"] for i in items]
assert len(ids) == len(set(ids)), "duplicate item ids"
os.makedirs(os.path.join(SP, "atm"), exist_ok=True)
json.dump(items, open(os.path.join(SP, "atm", "items.json"), "w"), ensure_ascii=False)
shutil.copy(os.path.join(D, "atm-bench/atm-bench.json"), os.path.join(SP, "atm", "atm-bench.json"))
qa = json.load(open(os.path.join(SP, "atm", "atm-bench.json")))
known = set(ids)
missing = sorted({e for q in qa for e in q["evidence_ids"] if e not in known})
print("items", len(items), {m: sum(1 for i in items if i["modality"] == m) for m in ("email", "image", "video")},
      "questions", len(qa), "evidence ids not in items:", len(missing), missing[:5])
sessions = [{"session_id": i["item_id"], "started_at": epoch(i["metadata"]["timestamp"]),
             "turns": [{"unit_id": i["item_id"] + "#0", "speaker": "email" if i["modality"] == "email" else i["modality"],
                        "text": i["text"], "at": epoch(i["metadata"]["timestamp"])}], "source_kind": "document"} for i in items]
sessions.sort(key=lambda s: s["started_at"])
questions = [{"qid": q["id"], "question": q["question"], "question_date": None, "answer": q["answer"], "type": q["qtype"],
              "evidence": {"session_ids": list(q["evidence_ids"]), "unit_ids": [e + "#0" for e in q["evidence_ids"]]}, "abstain": False}
             for q in qa]
with open(os.path.join(SP, "atm", "atm-bench.doc.jsonl"), "w") as f:
    f.write(json.dumps({"task_id": "atm-bench", "library": "atm", "sessions": sessions, "questions": questions}, ensure_ascii=False) + "\n")
print("wrote atm/items.json, atm/atm-bench.json, atm/atm-bench.doc.jsonl")

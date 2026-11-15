"""Side process: embed the C (header160) and D (chunk400) units on MPS and save the arrays proxy_chunking.py reuses."""
import json, re, os, sys, time, numpy as np
SP = os.path.dirname(os.path.abspath(__file__)); os.chdir(SP)
ITEMS = json.load(open("atm/items-san.json"))
def chunk_turn(text, max_words=160):  # verbatim port of mediafind.textsource.chunk_turn
    text = " ".join(text.split())
    if len(text.split()) <= max_words: return [text]
    chunks, pending = [], []
    for sentence in re.split(r"(?<=[.!?。！？])\s+", text):
        words = sentence.split()
        if pending and len(pending) + len(words) > max_words: chunks.append(" ".join(pending)); pending = []
        while len(words) > max_words: chunks.append(" ".join(words[:max_words])); words = words[max_words:]
        pending.extend(words)
    if pending: chunks.append(" ".join(pending))
    return chunks
def header(item):
    n = 4 if item["modality"] in ("image", "video") else 2
    return " ".join(" ".join(l.split()) for l in item["text"].split("\n")[:n])

# ---- units per variant: list of (item_id, text)
def units_chunk(n, with_header=False):
    out = []
    for i in ITEMS:
        cs = chunk_turn(i["text"], n); h = header(i)
        for j, c in enumerate(cs): out.append((i["item_id"], (h + " " + c) if (with_header and j > 0) else c))
    return out

import torch
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-m3", device="mps"); m.max_seq_length = 1024
for tag, units in (("header160", units_chunk(160, True)), ("chunk400", units_chunk(400))):
    p = f"runs/proxy_{tag}.npy"
    if os.path.exists(p): print("exists", p, flush=True); continue
    texts = [t for _, t in units]; done = False
    for bs in (16, 4):
        try:
            t0 = time.time(); v = m.encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            np.save(p + ".tmp.npy", v.astype(np.float32)); os.replace(p + ".tmp.npy", p); print(f"saved {p} {v.shape} in {time.time()-t0:.0f}s bs {bs}", flush=True); done = True; break
        except RuntimeError as e:
            if "out of memory" not in str(e).lower(): raise
            print("OOM bs", bs, flush=True); torch.mps.empty_cache(); time.sleep(5)
    if not done: print("GAVE UP on", tag, flush=True)
print("side done", flush=True)

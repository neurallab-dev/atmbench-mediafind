"""Step-1 proxy for the document-chunking lever (ATM-Bench, sanitized items). $0, local.
Variants of the text unit the first stage scores:
  A chunk160  = the product's current chunking; dense vectors READ FROM THE STORE (data_san), BM25 proxy over the same chunks
  B whole     = one unit per item (bge-m3 reads 8192 tokens; items are <= ~2,300)
  C header160 = chunk160 with the item's header lines (ID/Type/Timestamp/Location) prepended to every chunk after the first
  D chunk400  = 400-word chunks, no header (maps directly onto the product's DOCUMENT_CHUNK_WORDS)
Per variant: dense-only, BM25-only and hybrid (RRF k=60 over the unit lists, item = its best unit) item rankings for the
original questions and for the rewritten queries; R@k, uncovered@25, where the 115 ids missed by the submitted fused list land.
Pre-stated gate (original questions, hybrid): best of B/C/D >= A + 1.0 at R@25 and R@10 >= A - 0.3.
"""
import json, re, math, os, sys, time, collections, sqlite3
import numpy as np
SP = os.path.dirname(os.path.abspath(__file__)); os.chdir(SP)
t_start = time.time()
def log(*a): print(f"[{time.time()-t_start:6.0f}s]", *a, flush=True)
ITEMS = json.load(open("atm/items-san.json")); by = {i["item_id"]: i for i in ITEMS}; ids = [i["item_id"] for i in ITEMS]
qa = json.load(open("atm/atm-bench.json")); byid = {q["id"]: q for q in qa}
key = json.load(open("atm/rw-san-key.json"))
rw_text = {byid[v]["question"]: k for k, v in key.items() if v in byid}  # original question -> rewritten query text
log("items", len(ITEMS), "questions", len(qa), "rewritten", len(rw_text))

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
UNITS = {"A": units_chunk(160), "B": [(i["item_id"], " ".join(i["text"].split())) for i in ITEMS], "C": units_chunk(160, True), "D": units_chunk(400)}
for k, u in UNITS.items(): log("variant", k, "units", len(u), "words p50", int(np.median([len(t.split()) for _, t in u])))

# ---- variant A dense vectors from the store (must line up with our re-chunking)
con = sqlite3.connect("data_san/memory/atm/index.sqlite3")
rows = con.execute("SELECT t.session_id, s.seg_index, s.text, s.embedding FROM segments s JOIN text_sessions t ON t.media_id = s.media_id ORDER BY t.session_id, s.seg_index").fetchall()
store = {}
for sid, j, text, blob in rows: store.setdefault(sid, []).append((j, text, np.frombuffer(blob, dtype=np.float32)))
VA, UA = [], []
mism = 0
for i in ITEMS:
    segs = sorted(store[i["item_id"]]); mine = chunk_turn(i["text"], 160)
    if len(segs) != len(mine) or any(s[1] != m for s, m in zip(segs, mine)): mism += 1
    for j, text, v in segs: VA.append(v); UA.append((i["item_id"], text))
VA = np.vstack(VA).astype(np.float32); log("store vectors", VA.shape, "norm p50", float(np.median(np.linalg.norm(VA, axis=1))), "items whose chunks differ from our re-chunk:", mism)
UNITS["A"] = UA  # use the store's exact chunk texts for A's BM25 too

# ---- embeddings for B, C, D (cached as .npy)
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Library/Application Support/MediaFind/models/huggingface"))
import torch
from sentence_transformers import SentenceTransformer
_model = {}
def model(dev):
    if dev not in _model:
        m = SentenceTransformer("BAAI/bge-m3", device=dev); m.max_seq_length = int(os.environ.get("PROXY_MAXLEN", "2048")); _model[dev] = m
    return _model[dev]
def encode(texts, tag):
    p = f"runs/proxy_{tag}.npy"
    if os.path.exists(p):
        v = np.load(p)
        if len(v) == len(texts): log("reusing", p, v.shape); return v
    for dev, bs in [(d, int(b)) for d, b in (x.split(":") for x in os.environ.get("PROXY_DEVICES", "mps:8,mps:2,cpu:8").split(","))]:
        try:
            t0 = time.time(); v = model(dev).encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            log(f"encoded {tag}: {len(texts)} texts on {dev} bs {bs} in {time.time()-t0:.0f}s"); np.save(p, v.astype(np.float32)); return v.astype(np.float32)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower(): raise
            log(f"OOM on {dev} bs {bs}; retrying"); torch.mps.empty_cache(); time.sleep(5)
    raise RuntimeError("encode failed")
Q = [q["question"] for q in qa]; QR = [rw_text[q] for q in Q]
VQ = encode(Q, "q_orig"); VQR = encode(QR, "q_rw")
VEC = {"A": VA}
for k in ("B", "C", "D"): VEC[k] = encode([t for _, t in UNITS[k]], {"B": "whole", "C": "header160", "D": "chunk400"}[k])

# ---- BM25 proxy (FTS5-like: k1 1.2, b 0.75; OR of content tokens)
STOP = set("the a an of in on at to for and or is was were be been i my me we our you it its this that these those with from by as about what when where which who how did do does during have has had there their them they he she his her".split())
tok = lambda s: [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1]
class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b; self.N = len(docs); self.dl = np.array([len(d) for d in docs], dtype=np.float32); self.avg = float(self.dl.mean())
        self.post = collections.defaultdict(list)
        for i, d in enumerate(docs):
            for t, f in collections.Counter(d).items(): self.post[t].append((i, f))
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}
    def scores(self, q):
        s = np.zeros(self.N, dtype=np.float32)
        for t in set(q):
            p = self.post.get(t)
            if not p: continue
            idf = self.idf[t]
            for i, f in p: s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg))
        return s
# ---- rankings
TOPU = 400
def item_order(scores, unit_items, top=TOPU):
    idx = np.argpartition(-scores, min(top, len(scores) - 1))[:top]; idx = idx[np.argsort(-scores[idx])]
    out, seen = [], set()
    for j in idx:
        if scores[j] <= 0 and len(out) > 0: break
        it = unit_items[j]
        if it not in seen: seen.add(it); out.append(it)
    return out
def rrf_units(lists, k=60):
    sc = collections.defaultdict(float)
    for lst in lists:
        for r, j in enumerate(lst): sc[j] += 1.0 / (k + r + 1)
    return sc
def rank_variant(vk, QV, qtexts):
    unit_items = [i for i, _ in UNITS[vk]]; V = VEC[vk]; bm = BM25([tok(t) for _, t in UNITS[vk]])
    S = QV @ V.T  # (nq, nunits)
    dense, lex, hyb = {}, {}, {}
    for qi, q in enumerate(Q):
        ds = S[qi]; bs = bm.scores(tok(qtexts[qi]))
        di = np.argsort(-ds)[:TOPU]; bi = np.argsort(-bs)[:TOPU]; bi = bi[bs[bi] > 0]
        dense[q] = item_order(ds, unit_items); lex[q] = item_order(bs, unit_items)
        sc = rrf_units([list(di), list(bi)])
        best = {}
        for j, s in sorted(sc.items(), key=lambda x: -x[1]):
            it = unit_items[j]
            if it not in best: best[it] = s
        hyb[q] = list(best)[:200]
    return dense, lex, hyb
def R(F, k): return 100 * sum(sum(e in F[q["question"]][:k] for e in q["evidence_ids"]) / len(q["evidence_ids"]) for q in qa) / len(qa)
cov = lambda F, q: set(q["evidence_ids"]) <= set(F[q["question"]][:25])
OLD = json.load(open("runs/san_fused.rank.json")); LEX100 = json.load(open("runs/san_full.lex.rank.json"))
miss = [(q["question"], e) for q in qa for e in q["evidence_ids"] if e not in OLD[q["question"]][:25]]
def band(r): return "absent" if r is None else "1-25" if r <= 25 else "26-50" if r <= 50 else "51-100" if r <= 100 else "101-200"
def line(name, F):
    unc = sum(not cov(F, q) for q in qa)
    c = collections.Counter(band(F[qq].index(e) + 1 if e in F[qq] else None) for qq, e in miss)
    return f"{name:26s} " + " ".join("R@%d %.2f" % (k, R(F, k)) for k in (1, 5, 10, 25, 50, 100)) + f" | unc@25 {unc:3d} | missed-115: " + " ".join(f"{b} {c.get(b,0)}" for b in ("1-25", "26-50", "51-100", "101-200", "absent"))
report = [line("product lex100 (real)", LEX100), line("submitted fused (real)", OLD)]
hyb_orig = {}
for vk in ("A", "B", "C", "D"):
    log("ranking variant", vk)
    for qset, QV, qtexts in (("orig", VQ, Q), ("rw", VQR, QR)):
        dense, lex, hyb = rank_variant(vk, QV, qtexts)
        if qset == "orig":
            hyb_orig[vk] = hyb
            report += [line(f"{vk} dense-only {qset}", dense), line(f"{vk} bm25-only {qset}", lex)]
        report.append(line(f"{vk} HYBRID {qset}", hyb))
        json.dump(hyb, open(f"runs/proxy_{vk}_{qset}.rank.json", "w"))
    print("\n".join(report[-4:]) if vk != "A" else "\n".join(report), flush=True)
A25, A10 = R(hyb_orig["A"], 25), R(hyb_orig["A"], 10)
best = max(("B", "C", "D"), key=lambda v: R(hyb_orig[v], 25))
ok = R(hyb_orig[best], 25) >= A25 + 1.0 and R(hyb_orig[best], 10) >= A10 - 0.3
summary = f"\nPROXY GATE {'PASS' if ok else 'FAIL'}: best variant {best} hybrid R@25 {R(hyb_orig[best],25):.2f} vs A {A25:.2f} (need >= +1.0), R@10 {R(hyb_orig[best],10):.2f} vs A {A10:.2f} (need >= -0.3)"
print(summary, flush=True)
open("runs/proxy_report.txt", "w").write("\n".join(report) + "\n" + summary + "\n")
log("done")

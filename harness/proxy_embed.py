"""Dense-only + hybrid proxy for a CANDIDATE EMBEDDER over the store's own chunk units (variant A of proxy_chunking). $0, local.
usage: python proxy_embed.py MODEL TAG [--dtype fp32|fp16|bf16] [--maxlen 1024] [--devices mps:16,mps:4,cpu:8] [--inst "Instruct: ...\nQuery:"]
Encodes the 24,818 stored chunks + the 1,013 original and rewritten queries (with and without the model's query instruction),
then reports dense-only / hybrid (RRF k=60 with the BM25 proxy, item = best unit) item rankings against the bge-m3 store vectors.
Pre-stated gates (original questions): dense-only R@25 >= 91.8 and R@10 >= 86.5; hybrid R@25 >= 92.82 and R@10 >= 85.82.
"""
import json, re, math, os, sys, time, collections, sqlite3, argparse
import numpy as np
SP = os.path.dirname(os.path.abspath(__file__)); os.chdir(SP)
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--dtype", default="fp32")
ap.add_argument("--maxlen", type=int, default=1024); ap.add_argument("--devices", default="mps:16,mps:4,cpu:8"); ap.add_argument("--inst", default=None)
ap.add_argument("--skip-units", action="store_true", help="only (re)rank from cached .npy files")
A = ap.parse_args()
t_start = time.time()
def log(*a): print(f"[{time.time()-t_start:6.0f}s]", *a, flush=True)
ITEMS = json.load(open("atm/items-san.json")); ids = [i["item_id"] for i in ITEMS]
qa = json.load(open("atm/atm-bench.json")); byid = {q["id"]: q for q in qa}
key = json.load(open("atm/rw-san-key.json"))
rw_text = {byid[v]["question"]: k for k, v in key.items() if v in byid}
Q = [q["question"] for q in qa]; QR = [rw_text[q] for q in Q]
log("items", len(ITEMS), "questions", len(qa), "model", A.model, "dtype", A.dtype)

# ---- the store's chunk units + bge-m3 vectors (baseline)
con = sqlite3.connect("data_san/memory/atm/index.sqlite3")
rows = con.execute("SELECT t.session_id, s.seg_index, s.text, s.embedding FROM segments s JOIN text_sessions t ON t.media_id = s.media_id ORDER BY t.session_id, s.seg_index").fetchall()
store = {}
for sid, j, text, blob in rows: store.setdefault(sid, []).append((j, text, np.frombuffer(blob, dtype=np.float32)))
VA, UA = [], []
for i in ITEMS:
    for j, text, v in sorted(store[i["item_id"]]): VA.append(v); UA.append((i["item_id"], text))
VA = np.vstack(VA).astype(np.float32); unit_items = [i for i, _ in UA]; texts = [t for _, t in UA]
log("store units", VA.shape)

# ---- candidate embedder
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Library/Application Support/MediaFind/models/huggingface"))
import torch
from sentence_transformers import SentenceTransformer
DT = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[A.dtype]
_model = {}
def model(dev):
    if dev not in _model:
        kw = {"model_kwargs": {"torch_dtype": DT}} if DT is not torch.float32 else {}
        m = SentenceTransformer(A.model, device=dev, tokenizer_kwargs={"padding_side": "left"}, **kw); m.max_seq_length = A.maxlen; _model[dev] = m
        log("loaded on", dev, "prompts", getattr(m, "prompts", None), "max_seq_length", m.max_seq_length)
    return _model[dev]
def encode(texts, tag, prompt=None, prompt_name=None):
    p = f"runs/pe_{A.tag}_{tag}.npy"
    if os.path.exists(p):
        v = np.load(p)
        if len(v) == len(texts): log("reusing", p, v.shape); return v
    if A.skip_units: raise SystemExit(f"missing {p}")
    for dev, bs in [(d, int(b)) for d, b in (x.split(":") for x in A.devices.split(","))]:
        try:
            t0 = time.time()
            v = model(dev).encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, prompt=prompt, prompt_name=prompt_name)
            log(f"encoded {tag}: {len(texts)} texts on {dev} bs {bs} in {time.time()-t0:.0f}s"); v = v.astype(np.float32); np.save(p, v); return v
        except RuntimeError as e:
            if "out of memory" not in str(e).lower(): raise
            log(f"OOM on {dev} bs {bs}; retrying: {str(e)[:300]}"); 
            try: torch.mps.empty_cache()
            except Exception: pass
            time.sleep(5)
    raise RuntimeError("encode failed")
# query instruction: the model's own "query" prompt if it has one, else --inst
m0 = model(A.devices.split(",")[0].split(":")[0]) if not A.skip_units else None
has_query_prompt = bool(m0 and getattr(m0, "prompts", None) and m0.prompts.get("query"))
inst_kw = {"prompt_name": "query"} if has_query_prompt else ({"prompt": A.inst} if A.inst else None)
log("instruction variant:", inst_kw)
VQ = {("orig", 0): encode(Q, "q_orig"), ("rw", 0): encode(QR, "q_rw")}
if inst_kw: VQ[("orig", 1)] = encode(Q, "q_orig_inst", **inst_kw); VQ[("rw", 1)] = encode(QR, "q_rw_inst", **inst_kw)
VU = encode(texts, "units")
VQ_BGE = np.load("runs/proxy_q_orig.npy")  # bge-m3 queries (proxy_chunking)
del _model; torch.mps.empty_cache() if torch.backends.mps.is_available() else None

# ---- BM25 proxy (as proxy_chunking.py)
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
bm = BM25([tok(t) for t in texts])
TOPU = 400
def item_order(scores, top=TOPU):
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
def fuse(lists):
    best = {}
    for j, s in sorted(rrf_units(lists).items(), key=lambda x: -x[1]):
        it = unit_items[j]
        if it not in best: best[it] = s
    return list(best)[:200]
BI = []
for qi in range(len(Q)):
    bs = bm.scores(tok(Q[qi])); bi = np.argsort(-bs)[:TOPU]; BI.append(list(bi[bs[bi] > 0]))
BIR = []
for qi in range(len(Q)):
    bs = bm.scores(tok(QR[qi])); bi = np.argsort(-bs)[:TOPU]; BIR.append(list(bi[bs[bi] > 0]))
def rank(QV, qset):
    S = QV @ VU.T; SB = (VQ_BGE @ VA.T) if qset == "orig" else None
    dense, hyb, hyb3 = {}, {}, {}
    for qi, q in enumerate(Q):
        ds = S[qi]; di = list(np.argsort(-ds)[:TOPU]); bi = BI[qi] if qset == "orig" else BIR[qi]
        dense[q] = item_order(ds); hyb[q] = fuse([di, bi])
        if SB is not None: hyb3[q] = fuse([di, list(np.argsort(-SB[qi])[:TOPU]), bi])
    return dense, hyb, hyb3
def R(F, k): return 100 * sum(sum(e in F[q["question"]][:k] for e in q["evidence_ids"]) / len(q["evidence_ids"]) for q in qa) / len(qa)
cov = lambda F, q: set(q["evidence_ids"]) <= set(F[q["question"]][:25])
OLD = json.load(open("runs/san_fused.rank.json"))
miss = [(q["question"], e) for q in qa for e in q["evidence_ids"] if e not in OLD[q["question"]][:25]]
def band(r): return "absent" if r is None else "1-25" if r <= 25 else "26-50" if r <= 50 else "51-100" if r <= 100 else "101-200"
def line(name, F):
    unc = sum(not cov(F, q) for q in qa)
    c = collections.Counter(band(F[qq].index(e) + 1 if e in F[qq] else None) for qq, e in miss)
    return f"{name:34s} " + " ".join("R@%d %.2f" % (k, R(F, k)) for k in (1, 5, 10, 25, 50, 100)) + f" | unc@25 {unc:3d} | missed-115: " + " ".join(f"{b} {c.get(b,0)}" for b in ("1-25", "26-50", "51-100", "101-200", "absent"))
report = [line("submitted fused (real)", OLD)]
# bge-m3 baseline (store vectors, same proxy)
SBq = VQ_BGE @ VA.T
bge_dense, bge_hyb = {}, {}
for qi, q in enumerate(Q):
    ds = SBq[qi]; bge_dense[q] = item_order(ds); bge_hyb[q] = fuse([list(np.argsort(-ds)[:TOPU]), BI[qi]])
report += [line("bge-m3 dense-only orig", bge_dense), line("bge-m3 HYBRID orig", bge_hyb)]
print("\n".join(report), flush=True)
res = {}
for (qset, inst), QVv in sorted(VQ.items()):
    dense, hyb, hyb3 = rank(QVv, qset); nm = f"{A.tag}{'+inst' if inst else ''} {qset}"
    res[(qset, inst)] = (dense, hyb, hyb3)
    rows_ = [line(f"{nm} dense-only", dense), line(f"{nm} HYBRID", hyb)] + ([line(f"{nm} HYBRID3 (+bge dense)", hyb3)] if hyb3 else [])
    report += rows_; print("\n".join(rows_), flush=True)
    json.dump(hyb, open(f"runs/pe_{A.tag}_{qset}{'_inst' if inst else ''}.hyb.rank.json", "w")); json.dump(dense, open(f"runs/pe_{A.tag}_{qset}{'_inst' if inst else ''}.dense.rank.json", "w"))
    if hyb3: json.dump(hyb3, open(f"runs/pe_{A.tag}_{qset}{'_inst' if inst else ''}.hyb3.rank.json", "w"))
# gates on the original questions, best of the instruction variants
cands = [(inst, res[("orig", inst)]) for inst in (0, 1) if ("orig", inst) in res]
bi_, (d, h, h3) = max(cands, key=lambda c: R(c[1][0], 25))
g1 = R(d, 25) >= 91.8 and R(d, 10) >= 86.5; g2 = R(h, 25) >= 92.82 and R(h, 10) >= 85.82
summary = (f"\nDENSE GATE {'PASS' if g1 else 'FAIL'}: {A.tag}{'+inst' if bi_ else ''} dense-only R@25 {R(d,25):.2f} (need >= 91.8; bge-m3 {R(bge_dense,25):.2f}) R@10 {R(d,10):.2f} (need >= 86.5; bge-m3 {R(bge_dense,10):.2f})"
           f"\nHYBRID GATE {'PASS' if g2 else 'FAIL'}: hybrid R@25 {R(h,25):.2f} (need >= 92.82; bge-m3 {R(bge_hyb,25):.2f}) R@10 {R(h,10):.2f} (need >= 85.82; bge-m3 {R(bge_hyb,10):.2f})")
print(summary, flush=True)
open(f"runs/pe_{A.tag}_report.txt", "w").write("\n".join(report) + "\n" + summary + "\n")
log("done")

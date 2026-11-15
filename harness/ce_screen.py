"""Offline reranker screen: rerank a rank json's top-N ids with a cross-encoder over the WHOLE item text; report paired R@k.
usage: ce_screen.py RANK.json OUT.json [model] [N] [max_length]   (env ITEMS_FILE, QA_FILE, CE_DEVICE=mps|cpu, CE_BS)
"""
import json, sys, time, os, torch
from sentence_transformers import CrossEncoder
SP = os.path.dirname(os.path.abspath(__file__))
rank = json.load(open(sys.argv[1])); out_p = sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "BAAI/bge-reranker-v2-m3"
N = int(sys.argv[4]) if len(sys.argv) > 4 else 50
ML = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
qa = {q["question"]: q for q in json.load(open(os.environ.get("QA_FILE", SP + "/atm/atm-bench.json")))}
items = {i["item_id"]: i["text"] for i in json.load(open(os.environ.get("ITEMS_FILE", SP + "/atm/items.json")))}
dev = os.environ.get("CE_DEVICE", "mps")
ce = CrossEncoder(model, max_length=ML, device=dev)
def r(R, qs, k): return sum(sum(e in R[q][:k] for e in qa[q]["evidence_ids"]) / len(qa[q]["evidence_ids"]) for q in qs) / len(qs)
out = {}; t0 = time.time(); qs = [q for q in qa if q in rank]
part_p = out_p + ".partial"
if os.path.exists(part_p):
    out = json.load(open(part_p)); print(f"[ce] resuming: {len(out)} queries already reranked", flush=True)
for n, q in enumerate(qs):
    if q in out: continue
    ids = [i for i in rank[q][:N] if i in items]
    if not ids: out[q] = []; continue
    for attempt in range(40):
        try:
            s = ce.predict([(q, items[i]) for i in ids], batch_size=int(os.environ.get("CE_BS", "8")), show_progress_bar=False); break
        except RuntimeError as e:
            if "out of memory" not in str(e): raise
            torch.mps.empty_cache(); time.sleep(10)
            if attempt == 12:
                print("[ce] MPS OOM persisted; moving to CPU", flush=True); ce.model.to("cpu"); ce._target_device = torch.device("cpu")
    else:
        raise RuntimeError("OOM persisted")
    order = sorted(range(len(ids)), key=lambda j: -float(s[j]))
    out[q] = [ids[j] for j in order] + rank[q][N:]
    if dev == "mps":
        torch.mps.empty_cache()  # the MPS caching allocator keeps every batch shape: ~9 GB/query otherwise
    if (n + 1) % 25 == 0:
        json.dump(out, open(part_p, "w"))
    if (n + 1) % 100 == 0:
        print(f"{n+1}/{len(qs)} {time.time()-t0:.0f}s  R@10 so far in {r(rank, qs[:n+1], 10):.3f} -> {r(out, qs[:n+1], 10):.3f}", flush=True)
json.dump(out, open(out_p, "w"))
if os.path.exists(part_p): os.remove(part_p)
print("model", model, "N", N, "n", len(qs), " ".join(f"R@{k} {r(rank, qs, k):.3f}->{r(out, qs, k):.3f}" for k in (1, 5, 10, 25, 50)), flush=True)

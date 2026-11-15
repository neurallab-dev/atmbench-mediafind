"""Offline reranker screen with a Qwen3-Reranker (causal LM, yes/no logit): rerank a rank json's top-N ids over the WHOLE item text.
usage: qwen_rerank.py RANK.json OUT.json [model] [N] [max_length]   (env ITEMS_FILE, QA_FILE, CE_DEVICE=mps|cpu, CE_BS, CE_LIMIT=<n questions>)
Same output contract as ce_screen.py (reranked top-N + the untouched tail; .partial resume; paired R@k report)."""
import json, sys, time, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
SP = os.path.dirname(os.path.abspath(__file__))
rank = json.load(open(sys.argv[1])); out_p = sys.argv[2]
model_id = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3-Reranker-0.6B"
N = int(sys.argv[4]) if len(sys.argv) > 4 else 50
ML = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
qa = {q["question"]: q for q in json.load(open(os.environ.get("QA_FILE", SP + "/atm/atm-bench.json")))}
items = {i["item_id"]: i["text"] for i in json.load(open(os.environ.get("ITEMS_FILE", SP + "/atm/items-san.json")))}
dev = os.environ.get("CE_DEVICE", "mps"); BS = int(os.environ.get("CE_BS", "8")); LIMIT = int(os.environ.get("CE_LIMIT", "0"))
DT = torch.float16 if dev == "mps" else torch.float32
tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=DT).to(dev).eval()
T_NO, T_YES = tok.convert_tokens_to_ids("no"), tok.convert_tokens_to_ids("yes")
PREFIX = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
P_T = tok.encode(PREFIX, add_special_tokens=False); S_T = tok.encode(SUFFIX, add_special_tokens=False)
INSTR = os.environ.get("CE_INSTR", "Given a web search query, retrieve relevant passages that answer the query")
def fmt(q, d): return f"<Instruct>: {INSTR}\n<Query>: {q}\n<Document>: {d}"
@torch.no_grad()
def score(pairs):
    enc = tok([fmt(q, d) for q, d in pairs], padding=False, truncation="longest_first", return_attention_mask=False, max_length=ML - len(P_T) - len(S_T))
    enc["input_ids"] = [P_T + e + S_T for e in enc["input_ids"]]
    enc = tok.pad(enc, padding=True, return_tensors="pt", max_length=ML)
    enc = {k: v.to(dev) for k, v in enc.items()}
    logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
    two = torch.stack([logits[:, T_NO], logits[:, T_YES]], dim=1).float()
    return torch.nn.functional.log_softmax(two, dim=1)[:, 1].exp().tolist()
def r(R, qs, k): return sum(sum(e in R[q][:k] for e in qa[q]["evidence_ids"]) / len(qa[q]["evidence_ids"]) for q in qs) / len(qs)
out = {}; t0 = time.time(); qs = [q for q in qa if q in rank]
if LIMIT: qs = qs[:LIMIT]
part_p = out_p + ".partial"
if os.path.exists(part_p):
    out = json.load(open(part_p)); print(f"[qr] resuming: {len(out)} queries already reranked", flush=True)
for n, q in enumerate(qs):
    if q in out: continue
    ids = [i for i in rank[q][:N] if i in items]
    if not ids: out[q] = []; continue
    order_len = sorted(range(len(ids)), key=lambda j: len(items[ids[j]]))  # batch similar lengths together
    s = [0.0] * len(ids)
    for b in range(0, len(ids), BS):
        js = order_len[b:b + BS]
        for attempt in range(40):
            try:
                sc = score([(q, items[ids[j]]) for j in js]); break
            except RuntimeError as e:
                if "out of memory" not in str(e).lower(): raise
                torch.mps.empty_cache(); time.sleep(10)
        else: raise RuntimeError("OOM persisted")
        for j, v in zip(js, sc): s[j] = v
    order = sorted(range(len(ids)), key=lambda j: -s[j])
    out[q] = [ids[j] for j in order] + rank[q][N:]
    if dev == "mps": torch.mps.empty_cache()
    if (n + 1) % 25 == 0: json.dump(out, open(part_p, "w"))
    if (n + 1) % 50 == 0 or n + 1 == len(qs):
        done = [x for x in qs[:n + 1] if x in out]
        print(f"[qr] {n+1}/{len(qs)} {time.time()-t0:.0f}s  R@1 {r(rank, done, 1):.3f}->{r(out, done, 1):.3f}  R@10 {r(rank, done, 10):.3f}->{r(out, done, 10):.3f}  R@25 {r(rank, done, 25):.3f}->{r(out, done, 25):.3f}", flush=True)
json.dump(out, open(out_p, "w"))
if os.path.exists(part_p): os.remove(part_p)
print(f"[qr] done {len(out)} q in {time.time()-t0:.0f}s -> {out_p}", flush=True)

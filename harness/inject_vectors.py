"""Build a copy of the sanitized memory store whose segment vectors come from a candidate embedder (the proxy's .npy over the
store's own chunks, same order as proxy_embed.py builds its units), so the product's retrieval path can run with
MEDIAFIND_EMBED=<model> and no re-ingest.  usage: inject_vectors.py MODEL TAG   -> data_TAG/"""
import json, os, shutil, sqlite3, sys, numpy as np
SP = os.path.dirname(os.path.abspath(__file__)); os.chdir(SP)
MODEL, TAG = sys.argv[1], sys.argv[2]
V = np.load(f"runs/pe_{TAG}_units.npy").astype(np.float32); print("vectors", V.shape)
ITEMS = json.load(open("atm/items-san.json"))
src = "data_san/memory/atm/index.sqlite3"; dst_dir = f"data_{TAG}/memory/atm"; os.makedirs(dst_dir, exist_ok=True)
sqlite3.connect(src).execute("PRAGMA wal_checkpoint(TRUNCATE)")
shutil.copy(src, f"{dst_dir}/index.sqlite3")
con = sqlite3.connect(f"{dst_dir}/index.sqlite3")
rows = con.execute("SELECT t.session_id, s.seg_index, s.id FROM segments s JOIN text_sessions t ON t.media_id = s.media_id").fetchall()
by = {}
for sid, j, segid in rows: by.setdefault(sid, []).append((j, segid))
order = [segid for i in ITEMS for j, segid in sorted(by[i["item_id"]])]
assert len(order) == len(V), (len(order), len(V))
con.executemany("UPDATE segments SET embedding = ?, dim = ?, embedding_model_id = ? WHERE id = ?", [(V[n].tobytes(), int(V.shape[1]), MODEL, segid) for n, segid in enumerate(order)])
con.commit()
print(con.execute("SELECT embedding_model_id, dim, count(*) FROM segments GROUP BY 1,2").fetchall())
m = json.load(open("data_san/ingested-atm.json")); m["embed"] = MODEL; m["injected_from"] = f"runs/pe_{TAG}_units.npy"
json.dump(m, open(f"data_{TAG}/ingested-atm.json", "w")); print("wrote", f"data_{TAG}")

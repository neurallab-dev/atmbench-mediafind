"""Add a 'MediaFind v2' row + § note to leaderboard.html (the original MediaFind row, † note and evidence stay untouched).

usage: apply_board_v2.py FORK_DIR EVIDENCE.json   (run on a checkout reset to upstream/main)
"""
import json, re, sys, shutil, os
fork, ev_p = sys.argv[1], sys.argv[2]
ev = json.load(open(ev_p)); m = ev["splits"]["main"]
qs, r10 = round(m["qs_percent"], 2), round(m["recall_at_10"], 2)
p = os.path.join(fork, "leaderboard.html"); s = open(p, encoding="utf-8").read()
assert "MediaFind v2" not in s, "v2 row already present"
assert "&#167;" not in s, "§ marker already in use"

row_re = re.compile(r"^(\s*)\{ type: 'RAG', harness: 'MediaFind', index_time: null, answer_model: 'Qwen3-VL-8B-Instruct \(OpenRouter\)'.*\},?$", re.M)
mm = row_re.search(s); assert mm and len(row_re.findall(s)) == 1, "original MediaFind row not found exactly once"
v2_row = (f"{mm.group(1)}{{ type: 'Memory', harness: 'MediaFind v2', index_time: null, "
          "answer_model: 'GPT-5.4 mini (OpenRouter) + DeepSeek V4.1 Flash map', "
          "memory_model: 'DeepSeek V4.1 Flash cards + DeepSeek V4 Flash Vision descriptions', "
          "retriever: 'bge-m3 + BM25 + bge-reranker-v2-m3 + LLM query rewrite', "
          f"qs: {qs:.2f}, recall: {r10:.2f}, link: 'https://mediafind.io', "
          "notes: { qs: '\\u00a7', answer_model: '\\u00a7', memory_model: '\\u00a7' } },")
s = s[:mm.end()] + "\n" + v2_row + s[mm.end():]

NOTE_EN = ("<strong>&#167;</strong> MediaFind v2 retrieval: MediaFind&#39;s on-device memory search (bge-m3 + BM25 lexical fusion) ranks "
           "the benchmark&#39;s captions and emails, bge-reranker-v2-m3 reranks the whole text of the top-50 items, and that ranking is "
           "reciprocal-rank fused with the lexical ranking of a DeepSeek V4.1 Flash rewrite of the question. Memory: each photo/video "
           "carries a memory card written by DeepSeek V4.1 Flash from its caption, the person&#39;s emails (a trip/event timeline "
           "extracted from all emails) and same-day items, plus a dense visual description of the raw photo or video frames by "
           "DeepSeek V4 Flash Vision, both via OpenRouter. Answering: a map-reduce over the top-25 items, in which DeepSeek V4.1 Flash "
           "labels the relevant items per batch of ten and GPT-5.4 mini (reasoning effort minimal) answers from the kept items&#39; full "
           "text, both via OpenRouter; QS uses gpt-5-mini. <a href=\"docs/mediafind-v2-atmbench.json\">Per-question score summary</a> "
           "(with the top-25 retrieved ids).")
NOTE_ZH = ("<strong>&#167;</strong> MediaFind v2 检索：MediaFind 端侧记忆检索（bge-m3 + BM25 融合）对同一批 caption 和邮件排序，"
           "bge-reranker-v2-m3 对 top-50 整条重排，再与 DeepSeek V4.1 Flash 改写后问题的词法排序做倒数排名融合。"
           "记忆：每张照片/视频附带一张由 DeepSeek V4.1 Flash 根据 caption、邮件（从全部邮件提取的行程/事件时间线）和同日条目生成的记忆卡片，"
           "以及 DeepSeek V4 Flash Vision 对原始照片/视频帧生成的细节描述（均经 OpenRouter）。"
           "回答：对 top-25 条目做 map-reduce：DeepSeek V4.1 Flash 每 10 条筛选相关条目，GPT-5.4 mini（reasoning effort minimal）据保留条目全文作答（均经 OpenRouter）；"
           "QS 使用 gpt-5-mini 评判。<a href=\"docs/mediafind-v2-atmbench.json\">逐题得分摘要</a>（含 top-25 检索 id）。")
anchor_en = '<a href="docs/mediafind-atmbench.json">Per-question score summary</a>.'
anchor_zh = '<a href="docs/mediafind-atmbench.json">逐题得分摘要</a>。'
assert s.count(anchor_en) == 2 and s.count(anchor_zh) == 1, (s.count(anchor_en), s.count(anchor_zh))
s = s.replace(anchor_en, anchor_en + " " + NOTE_EN).replace(anchor_zh, anchor_zh + NOTE_ZH)
open(p, "w", encoding="utf-8").write(s)
shutil.copy(ev_p, os.path.join(fork, "docs", "mediafind-v2-atmbench.json"))
print(f"v2 row added (qs {qs:.2f}, recall {r10:.2f}); § notes inserted (EN x2, ZH x1); evidence -> docs/mediafind-v2-atmbench.json")

"""Rewrite the MediaFind row and its † footnote in leaderboard.html from the evidence json.

usage: apply_board.py FORK_DIR EVIDENCE.json
"""
import json, re, sys, shutil, os
fork, ev_p = sys.argv[1], sys.argv[2]
ev = json.load(open(ev_p)); m = ev["splits"]["main"]
qs, r10 = round(m["qs_percent"], 2), round(m["recall_at_10"], 2)
VISION = os.environ.get("VISION") == "1"; REWRITE = os.environ.get("QUERY_REWRITE") == "1"
p = os.path.join(fork, "leaderboard.html"); s = open(p, encoding="utf-8").read()

row_re = re.compile(r"^(\s*)\{ type: 'RAG', harness: 'MediaFind',.*\},?$", re.M)
assert len(row_re.findall(s)) == 1, "expected exactly one MediaFind row"
new_row = ("{indent}{{ type: 'RAG', harness: 'MediaFind', index_time: null, "
           "answer_model: 'GPT-5.4 mini (OpenRouter) + DeepSeek V4.1 Flash map', "
           "memory_model: '" + ("DeepSeek V4.1 Flash cards + DeepSeek V4 Flash Vision descriptions" if VISION else "DeepSeek V4.1 Flash memory cards") + "', "
           "retriever: '" + ("bge-m3 + BM25 + bge-reranker-v2-m3 + LLM query rewrite" if REWRITE else "bge-m3 + BM25 + bge-reranker-v2-m3") + "', qs: {qs:.2f}, recall: {r10:.2f}, "
           "link: 'https://mediafind.io', notes: {{ qs: '\\u2020', answer_model: '\\u2020', memory_model: '\\u2020' }} }},")
s = row_re.sub(lambda mm: new_row.format(indent=mm.group(1), qs=qs, r10=r10), s)

old_en = ("<strong>&#8224;</strong> MediaFind runs the official MMRAG pipeline with its own on-device ranking over the same captions and emails; "
          "Qwen3-VL-8B answers via OpenRouter (hosted, not the local FP8 copy). On that setup the official ATM-RAG retriever scores QS 48.4 / R@10 68.1. "
          "<a href=\"docs/mediafind-atmbench.json\">Per-question score summary</a>.")
new_en = ("<strong>&#8224;</strong> MediaFind ranks the benchmark's own captions and emails with its on-device memory search (bge-m3 + BM25, "
          "whole-item bge-reranker-v2-m3 over the top-50" + (", fused with the ranking of a DeepSeek V4.1 Flash rewrite of the question" if REWRITE else "") + "); each photo/video is extended with a memory card written by DeepSeek V4.1 Flash "
          "(OpenRouter) from its caption, the person's emails and same-day items" + (" and a dense visual description of the raw photo/video by DeepSeek V4 Flash Vision (OpenRouter)" if VISION else "") + "; answers come from a map-reduce over the top-25 items "
          "(DeepSeek V4.1 Flash relevance map, GPT-5.4 mini reduce, both hosted via OpenRouter); QS uses gpt-5-mini. "
          "<a href=\"docs/mediafind-atmbench.json\">Per-question score summary</a>.")
assert s.count(old_en) >= 1, "EN footnote not found"
s = s.replace(old_en, new_en)
old_zh_re = re.compile(r"<strong>&#8224;</strong> MediaFind 使用官方 MMRAG 流程[^<]*<a href=\"docs/mediafind-atmbench.json\">逐题得分摘要</a>。")
assert len(old_zh_re.findall(s)) == 1, "ZH footnote not found"
new_zh = ("<strong>&#8224;</strong> MediaFind 用自己的端侧记忆检索（bge-m3 + BM25，top-50 整条 bge-reranker-v2-m3 重排" + ("，并与 DeepSeek V4.1 Flash 改写后的查询排序融合" if REWRITE else "") + "）对同一批 caption 和邮件排序；"
          "每张照片/视频附带一张由 DeepSeek V4.1 Flash（OpenRouter）根据 caption、邮件和同日条目生成的记忆卡片" + ("，以及 DeepSeek V4 Flash Vision（OpenRouter）对原始照片/视频生成的细节描述" if VISION else "") + "；"
          "回答为 top-25 条目的 map-reduce（DeepSeek V4.1 Flash 做相关性筛选，GPT-5.4 mini 作答，均为 OpenRouter 托管）；QS 使用 gpt-5-mini 评判。"
          "<a href=\"docs/mediafind-atmbench.json\">逐题得分摘要</a>。")
s = old_zh_re.sub(new_zh, s)
open(p, "w", encoding="utf-8").write(s)
shutil.copy(ev_p, os.path.join(fork, "docs", "mediafind-atmbench.json"))
print(f"row -> qs {qs:.2f} recall {r10:.2f}; footnotes updated; evidence copied")

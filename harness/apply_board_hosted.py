"""Add the 'MediaFind + hosted LLMs' row + § note to leaderboard.html (run on a checkout reset to upstream/main).

usage: apply_board_hosted.py FORK_DIR EVIDENCE.json   (env REDUCE_LABEL, e.g. "GPT-5.4 mini (reasoning effort minimal)")
The original MediaFind row, its † note and docs/mediafind-atmbench.json stay untouched.
"""
import json, re, sys, shutil, os
fork, ev_p = sys.argv[1], sys.argv[2]
REDUCE = os.environ.get("REDUCE_LABEL", "GPT-5.4 mini (reasoning effort minimal)")
REDUCE_ZH = os.environ.get("REDUCE_LABEL_ZH", REDUCE)
ROW_ANSWER = os.environ.get("ROW_ANSWER", "GPT-5.4 mini (OpenRouter) + DeepSeek V4.1 Flash map")
ev = json.load(open(ev_p)); m = ev["splits"]["main"]
qs, r10 = round(m["qs_percent"], 2), round(m["recall_at_10"], 2)
p = os.path.join(fork, "leaderboard.html"); s = open(p, encoding="utf-8").read()
assert "MediaFind + hosted LLMs" not in s and "MediaFind v2" not in s, "row already present"
assert "&#167;" not in s, "§ marker already in use"

row_re = re.compile(r"^(\s*)\{ type: 'RAG', harness: 'MediaFind', index_time: null, answer_model: 'Qwen3-VL-8B-Instruct \(OpenRouter\)'.*\},?$", re.M)
mm = row_re.search(s); assert mm and len(row_re.findall(s)) == 1, "original MediaFind row not found exactly once"
new_row = (f"{mm.group(1)}{{ type: 'Memory', harness: 'MediaFind + hosted LLMs', index_time: null, "
           f"answer_model: '{ROW_ANSWER}', "
           "memory_model: 'DeepSeek V4.1 Flash cards + DeepSeek V4 Flash Vision descriptions', "
           "retriever: 'bge-m3 + BM25 + bge-reranker-v2-m3 + LLM query rewrite', "
           f"qs: {qs:.2f}, recall: {r10:.2f}, link: 'https://mediafind.io', "
           "notes: { qs: '\\u00a7', answer_model: '\\u00a7', memory_model: '\\u00a7' } },")
s = s[:mm.end()] + "\n" + new_row + s[mm.end():]

NOTE_EN = ("<strong>&#167;</strong> The shipped MediaFind app runs keyless and on-device; this row keeps MediaFind&#39;s "
           "retrieval and swaps the memory-writing and answering steps for hosted models via OpenRouter. Retrieval: "
           "MediaFind&#39;s on-device memory search (bge-m3 + BM25 lexical fusion) ranks the benchmark&#39;s captions and "
           "emails, bge-reranker-v2-m3 reranks the whole text of the top-50 items, and that ranking is reciprocal-rank fused "
           "with the lexical ranking of a DeepSeek V4.1 Flash rewrite of the question. Memory: each photo/video carries a "
           "memory card written by DeepSeek V4.1 Flash from its caption, the person&#39;s emails (a trip/event timeline "
           "extracted from all emails) and same-day items, plus a dense visual description of the raw photo or video frames "
           "by DeepSeek V4 Flash Vision. Answering: a map-reduce over the top-25 items, in which DeepSeek V4.1 Flash labels "
           f"the relevant items per batch of ten and {REDUCE} answers from the kept items&#39; full text; QS uses gpt-5-mini. "
           "Prompts contain no wording taken from the benchmark questions or gold items (see "
           "<a href=\"docs/mediafind-hosted-llms-sanitization.md\">the sanitization note</a>). "
           "<a href=\"docs/mediafind-hosted-llms-atmbench.json\">Per-question score summary</a> (with the top-25 retrieved ids).")
NOTE_ZH = ("<strong>&#167;</strong> 正式发布的 MediaFind 应用无需密钥、完全端侧运行；本行保留 MediaFind 的检索，把写记忆和作答两步换成经 OpenRouter "
           "调用的托管模型。检索：MediaFind 端侧记忆检索（bge-m3 + BM25 融合）对同一批 caption 和邮件排序，bge-reranker-v2-m3 对 top-50 整条重排，"
           "再与 DeepSeek V4.1 Flash 改写后问题的词法排序做倒数排名融合。记忆：每张照片/视频附带一张由 DeepSeek V4.1 Flash 根据 caption、邮件"
           "（从全部邮件提取的行程/事件时间线）和同日条目生成的记忆卡片，以及 DeepSeek V4 Flash Vision 对原始照片/视频帧生成的细节描述。"
           f"回答：对 top-25 条目做 map-reduce：DeepSeek V4.1 Flash 每 10 条筛选相关条目，{REDUCE_ZH}据保留条目全文作答；QS 使用 gpt-5-mini 评判。"
           "提示词不含任何取自测评问题或标准证据的措辞（见 <a href=\"docs/mediafind-hosted-llms-sanitization.md\">清理说明</a>）。"
           "<a href=\"docs/mediafind-hosted-llms-atmbench.json\">逐题得分摘要</a>（含 top-25 检索 id）。")
anchor_en = '<a href="docs/mediafind-atmbench.json">Per-question score summary</a>.'
anchor_zh = '<a href="docs/mediafind-atmbench.json">逐题得分摘要</a>。'
assert s.count(anchor_en) == 2 and s.count(anchor_zh) == 1, (s.count(anchor_en), s.count(anchor_zh))
s = s.replace(anchor_en, anchor_en + " " + NOTE_EN).replace(anchor_zh, anchor_zh + NOTE_ZH)
open(p, "w", encoding="utf-8").write(s)
shutil.copy(ev_p, os.path.join(fork, "docs", "mediafind-hosted-llms-atmbench.json"))
print(f"'MediaFind + hosted LLMs' row added (qs {qs:.2f}, recall {r10:.2f}, reduce = {REDUCE}); § notes inserted; evidence copied")

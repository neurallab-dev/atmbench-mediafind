Updates the MediaFind row on the ATM-Bench (main) board from QS 56.10 / R@10 77.85 to **QS QS_VALUE / R@10 R10_VALUE** (1,013 questions, gpt-5-mini judge with reasoning effort minimal, 0 answer failures, 0 judge failures), and replaces `docs/mediafind-atmbench.json` with the new per-question score summary (id, qtype, score, retrieved top-10 ids; no dataset text).

What changed since the previous row:

- **Memory cards.** Every photo and video is extended with a short card (event, place, when, what, keywords) written by DeepSeek V4.1 Flash via OpenRouter from the item's caption, the person's emails (a trip/event timeline extracted from all emails, plus the emails within ±10 days that name the photo's place) and the other items taken the same day. VISION_LINE
- **Retrieval.** MediaFind's memory search (bge-m3 + BM25 lexical fusion) ranks the benchmark's own captions and emails, then bge-reranker-v2-m3 reranks the whole text of the top-50 items. Full-set recall: R@1 R1_VALUE, R@5 R5_VALUE, R@10 R10_VALUE, R@25 R25_VALUE.
- **Answering.** Map-reduce over the top-25 items: DeepSeek V4.1 Flash labels relevant items per batch of ten, GPT-5.4 mini (reasoning effort minimal) answers from the kept items' full text; both hosted via OpenRouter. Per type: number NUM_VALUE, list_recall LIST_VALUE, open_end OE_VALUE.

The `answer_model` and `memory_model` columns name the hosted models, as the Memexa v2 row does; the † footnote (EN and ZH) describes the setup. `node tools/verify_leaderboard.mjs` passes.

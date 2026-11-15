## Summary

Records the ATM-Bench first-place attempt in `docs/MEMORY.md` §9 as **N11**: LLM memory cards for every photo/video (cards from captions + emails + same-day items), the whole-item bge-reranker-v2-m3 rerank, a hosted map-reduce answerer (DeepSeek V4.1 Flash map, GPT-5.4 mini reduce), the full-set numbers, the Oracle ceiling, every lever that was tried and rejected, and the two reboots that wiped `/private/tmp`.

Result on the board: MediaFind QS 56.10 / R@10 77.85 → **QS_VALUE / R10_VALUE** (second place; Memexa v2 76.68 / 92.32). BOARD_PR_LINE

Docs only; the product's keyless indexing and search are unchanged. Artifacts and scripts live under `~/.cache/mediafind-eval/atm-first/` (outside `/private/tmp`).

## Test plan

- [x] `make sync` (docs-only change; the fast suite is unaffected)
- [x] Numbers cross-checked against the eval dirs' `atm_openai_gpt-5-mini_summary.json` and the rank files

🤖 Generated with [Claude Code](https://claude.com/claude-code)

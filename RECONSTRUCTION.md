# How this harness was reconstructed (2026-11-15)

The original harness directory (`~/.cache/mediafind-eval/atm-first/`) was deleted on 2026-11-08 together with every
prediction, judge output, rank file and request cache in it. The scripts in `harness/` were rebuilt from the Claude
Code session logs that created and edited them (`recon/`):

1. `extract_events.py` collects, from 38 session logs, every tool call that wrote a file under the harness directory or
   an earlier session scratchpad: Write tool inputs and shell heredocs (`cat > file <<EOF`) are exact contents.
2. `replay.py` replays them in timestamp order into a mirror directory, together with the in-place edits made by other
   commands: `sed -i`, `sed ... src > dst`, `cp`, `mv`, and inline Python patch scripts (run sandboxed: no network, no
   subprocess, no writes outside the mirror). The Python patches assert the old text before replacing it; none of the
   patches to a harness script failed its assertion. Every intermediate version of every file is kept.
3. `verify2.py` re-checks every read the sessions made of a harness file (`grep -n` lines and `sed -n A,Bp` blocks
   printed in the logs) against the rebuilt version of that file at that moment: 129 checks match, 0 real mismatches
   (5 flagged lines are timestamps that look like `N:text` and one output that was cut at 200 characters).

What the replay cannot prove is the absence of an edit made in a form it does not parse; a scan for other forms
(`python -c` writes, `echo`/`printf` redirects, `perl -i`) found none touching a harness file.

## Code the harness runs against (pinned)

- MediaFind retrieval (`mfexp.py` imports the product's memory search from a worktree, run with the main checkout's venv):
  - bge-m3 ingest, lexical rankings and the rewritten-query rankings (2026-10-24): **6c783997** (main at 2026-10-23 23:56 UTC;
    the worktree carried one docs-only commit on top).
  - Qwen3-Embedding-4B rankings (2026-10-28/29): **bfacc365** (main at 2026-10-27 03:02 UTC; one docs-only commit on top).
  MediaFind's source is not public; its outputs (rank files) are what the published bundle carries for this stage.
- Official evaluator: JingbiaoMei/ATM-Bench at **ef4e5df** (`memqa/utils/evaluator/evaluate_qa.py`), judge gpt-5-mini via OpenRouter.

## The frozen configuration of the QS 80.69 row

| type | pool | relevance map | reduce | rank prior | post-processing | driver |
|---|---|---|---|---|---|---|
| open_end | `san_fused` = RRF k=10 (depth 100) of the bge-m3 whole-item rerank list (top-50, bge-reranker-v2-m3) and the bge-m3 lexical list of the DeepSeek V4.1 Flash rewrite | deepseek/deepseek-v4.1-flash, batches of 10 | openai/gpt-5.4-mini, effort minimal, never-abstain prompt (BEST_GUESS=1) | none | none | `run_full.sh` → `mf_answer_v2.py` |
| list_recall | `san_fused_e_q4b_w15` = RRF k=10 (depth 100) of the Qwen3-Embedding-4B lexical lists of three rewrites (DeepSeek V4.1 Flash, GPT-5.4 mini, GPT-5.4), the bge-m3 rerank list (weight 1.5) and the 4B rerank list (top-50) | deepseek/deepseek-v4.1-flash, all 25 in one call | as above | top-10 added (K_LIST=10) | ids kept only if in the pool | `run_full_b25.sh` → `mf_answer_r2.py` |
| number | `san_fused_e_q4b_w15` | anthropic/claude-haiku-5.5, batches of 10 | as above | none | `ans_format.py` | `run_number_full.sh` → `mf_answer_v2.py` |

Fallback model for a failed reduce: anthropic/claude-sonnet-5.5. Answer depth 25 for every type.

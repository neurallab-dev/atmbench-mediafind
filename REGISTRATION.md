# Pre-registration of the re-run (written 2026-11-15, before any paid stage ran)

- **What runs:** `harness/rerun_frozen.sh`, once, end to end: the frozen configuration in RECONSTRUCTION.md, from the
  released dataset (Jingbiao/ATM-Bench at 78e826dc) to one prediction file of 1,013 answers.
- **What may be repeated:** only a stage whose output is missing or incomplete (a provider error, a crash, a stopped
  run). The LLM request cache makes a repeated call return the cached response, so a repeat never redraws an answer
  that already exists. Cards and visual descriptions run two passes to fill provider gaps, as in October.
- **Judge:** the official evaluator (JingbiaoMei/ATM-Bench at ef4e5df), gpt-5-mini via OpenRouter, reasoning effort
  minimal, run **once** on the assembled file. No re-judging and no second answer pass for a better draw.
- **Reported result:** whatever this run scores replaces the QS 80.69 row, higher or lower. Its prediction file with
  SHA-256, the judge output and this harness are published with it.
- **Not changed:** no prompt, model, pool, fusion weight, depth, routing rule or post-processing step. Changes for the
  re-run are paths only (see the git history), and the regenerated LLM-written item text (timeline, cards, visual
  descriptions, rewrites) is a fresh sample of the same prompts and models.

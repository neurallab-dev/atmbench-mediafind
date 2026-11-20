# Answer-step screening, 2026-11-20: results

Phase 1 ran on 2026-11-20 between 19:37 and 19:50 UTC, as planned in `PLAN.md` (committed before the first call). Every
variant replayed the re-run's 288 relevance-map calls from the cache and made 96 new reduce calls. There were no
failed answers and no fallbacks, and each variant was judged once.

| variant | reduce model | open_end on the 96 subset questions | against the baseline (gained / lost) | harness cost estimate |
|---|---|---|---|---|
| baseline | the re-run's answers (GPT-5.4 mini, effort minimal) | 78.12 | | |
| S1 | openai/gpt-6.2-astra, effort low | 76.04 | 4 / 6 | $4.22 |
| S2 | anthropic/claude-opus-5.5, no reasoning | **80.21** | 5 / 3 | $3.13 |
| S3 | openai/gpt-6.2-sol, effort low | 73.96 | 3 / 7 | $1.05 |

**Outcome: the stop rule applies.** The best variant (S2) is 2.09 points above the baseline. The plan required 3
points before any further phase. Phases 2 to 4 were not run, and the row stays at the re-run's QS 77.85. A net of
two questions out of 96 is also within what a different draw does on this subset.

OpenRouter spend for the phase, from the account balance: $8.48. Logs are in `logs/`.

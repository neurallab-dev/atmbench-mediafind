# Frozen re-run of the "MediaFind + hosted LLMs" row (2026-11-15 to 2026-11-17)

One end-to-end run of the configuration frozen in `RECONSTRUCTION.md`, driven by `harness/rerun_frozen.sh` under the
conditions written down beforehand in `REGISTRATION.md`. It regenerates everything an LLM writes (trip timeline,
memory cards, visual descriptions, query rewrites, answers) and was judged once.

## Result

| answer set | QS | number | list_recall | open_end | R@1 | R@5 | R@10 | R@25 |
|---|---|---|---|---|---|---|---|---|
| this re-run (judged 2026-11-17, 06:56-07:13 UTC) | **77.85** | 83.06 | 75.29 | 74.90 | 53.74 | 85.19 | 90.51 | 94.69 |
| the QS 80.69 row (assembled 2026-10-29 from three selection-time runs) | 80.69 | 85.00 | 80.15 | 77.82 | 55.96 | 85.23 | 90.73 | 94.84 |

1,013 predictions, 0 empty answers, 0 answer failures, 0 judge failures (514 open-ended judgments, all new, none
retried). Per-type figures are the official evaluator's `atm` summary. Retrieval is measured on the ranking each
question type's answers were drawn from (`ranks/san_fused_typecond2.rank.json`).

Prediction file: `predictions/mediafind-hosted-llms-rerun.jsonl` in the release asset,
SHA-256 `5a88d4ea3283cac116b2243e484a956ea9cc7b1240ade149619d71d86a053a32`. `SHA256SUMS` in this folder covers every
file in the asset.

## Pins

| what | pin |
|---|---|
| dataset | `Jingbiao/ATM-Bench` (Hugging Face) at `78e826dc07e97466b2f54443831ef9a83ab8b27c`; question file `atm-bench/atm-bench.json`, SHA-256 `ab6eaa9df62fb4162e0f5eecd98768a7e3ae721e32d2db2cf227ff41e3295762` (the judge read a byte-identical copy) |
| evaluator | `JingbiaoMei/ATM-Bench` at `ef4e5dff1a47ec71213a06e359f02753defa8fb1`, `memqa/utils/evaluator/evaluate_qa.py` |
| judge command | `harness/judge.sh`: `evaluate_qa.py --ground-truth atm/atm-bench.json --predictions ans/mediafind-hosted-llms-rerun.jsonl --output-dir ans/mediafind-hosted-llms-rerun.eval --metrics em atm --judge-provider openai --judge-model openai/gpt-5-mini --judge-reasoning-effort minimal --max-workers 6`, with `OPENAI_BASE_URL=https://openrouter.ai/api/v1` |
| first-stage ranking | MediaFind `6c783997` for the bge-m3 ingest, lexical and rewrite rankings; `bfacc365` for the Qwen3-Embedding-4B rankings. The MediaFind repository is private, so its ranking outputs are published instead (`ranks/`) |
| harness | this repository at tag `rerun-2026-11-17` |
| driver | `harness/rerun_frozen.sh`; Python 3.12 for the harness, the MediaFind virtual environment for the first stage |

## What happened, in order

- **2026-11-15, 05:58-15:18 UTC.** Timeline (1,517 records), memory cards (4,292) and visual descriptions (4,292)
  regenerated from the raw dataset, none from cache. Local rankings: bge-m3 lexical R@10 85.8 / R@25 90.8
  (October: 85.3 / 90.9), after the top-50 rerank 87.2 / 91.5, fused with the DeepSeek rewrite 89.9 / 93.9
  (October: 89.7 / 93.5). DeepSeek V4.1 Flash and GPT-5.4 mini rewrite sets complete with 0 failed calls. Qwen3-Embedding-4B
  encoding and rankings on the question and both rewrite sets.
- **2026-11-15, 15:18 UTC: the OpenRouter account ran out of credits** during the GPT-5.4 rewrite set. 366 of
  1,013 rewrites failed with HTTP 402, and `rewrite_queries.py` kept the original question for those, as it does
  for any failed call. The driver's balance check had read the API key's spending limit instead of the account's
  credits, so it went on: the Qwen3-Embedding-4B ranking of that incomplete set, the two answer pools built on it,
  and all three answer stages, in which every call failed with HTTP 402 except the first 17 open-ended relevance-map
  calls. The driver then stopped on the failed answer lines. None of those outputs was judged or used: they were moved aside (the incomplete rewrite set and its
  ranking, both pools, the three answer files), and the driver was changed to check the account's credits and to
  abort on any failed rewrite. A failed call is never written to the request cache, so nothing from these calls can
  reach a later output.
- **2026-11-17, 06:18 UTC: resumed** after credits were added. The 366 missing GPT-5.4 rewrites were requested
  (the 647 that had succeeded on 2026-11-15 came from the request cache, as `REGISTRATION.md` allows for an
  incomplete stage, and so did the 17 open-ended map calls, which use the unaffected open-ended pool), then the 4B ranking of the complete set, both pools, the 514 open-ended, 139 list and 360
  number answers, assembly, and the single judge pass.

Driver changes between the first attempt and the resume, none of which touches a pipeline setting: the local 4B
stages run before the GPT-5.4 rewrite set (order only), the shell drivers' execute bit, the credit check, and the
abort on failed rewrites. Every stage the driver repeated was missing or incomplete.

## Cost

OpenRouter, from the account balance before and after: **$16.59** ($7.56 on 2026-11-15, $9.03 on 2026-11-17).
The harness's own per-stage estimates: timeline $0.25, memory cards $2.09, visual descriptions $1.30, rewrites
$0.04 (DeepSeek) + $1.08 (GPT-5.4 mini) + about $4.22 (GPT-5.4, both days), answers $3.38 (open-ended) + $1.67
(list) + $2.18 (number), judge about $0.16.

## Reading the gap

Retrieval reproduces within 0.3 points at R@5, R@10 and R@25 even though the item text was written afresh; R@1 is
2.2 points lower. The 2.84-point QS gap is in answering:

- number: the frozen setting was adopted because its pass scored 306 of 360 (a pass on the earlier pool scored
  302); this run scored 299.
- list_recall: the frozen setting was adopted after full passes that scored 77.3, 80.2 and 76.7, and it was the
  80.2 one; this run scored 75.3.
- open_end: the row's answers came from a run judged 77.8, and three other full open-ended passes on nearby
  settings were judged 76.1, 74.9 and 77.0; this run was judged 74.9, the bottom of that range.

For number and list questions a fresh draw of a setting adopted for its high score is expected to land lower, and
open-ended scores move by about 1.5 points between passes. One run cannot separate that selection effect from
ordinary generation and judge variance, and no second run was made, so that there is no choice between draws.

## Publication edits

Before this repository was made public on 2026-11-17, its history was rewritten to (a) shorten a default path that
named the original machine's private working directory (`<private dir>/eval/atmbench` became `atmbench`; the re-run
sets these paths through environment variables, so no run behaves differently) and (b) drop one internal draft
document that is not part of the harness. Commit hashes from before that rewrite do not exist in this repository.

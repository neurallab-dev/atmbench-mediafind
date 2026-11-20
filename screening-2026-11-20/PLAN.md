# Answer-step screening, 2026-11-20 (written before any call is made)

Goal: find out whether a stronger answer step lifts the "MediaFind + hosted LLMs" row past QS 81.71 (Graphon V1 with
GPT-6 Astra). The frozen re-run of 2026-11-17 (`rerun-2026-11-17/RERUN.md`, QS 77.85) is the starting point, and
everything upstream of the answer step stays exactly as published there: the item text, the rankings and pools, and
the relevance-map decisions, which are replayed from the request cache.

Why the answer step: on the re-run's open-ended questions, the map kept every gold item for 419 questions and the
answer was still judged wrong on 70 of them; on 54 more, the map dropped a gold item. Per-question scores published on
the board show Graphon answering more of both groups (86.6 vs 83.3, and 61.1 vs 35.2), with lower retrieval recall
than ours (R@10 89.78 vs 90.51).

Screening set: the fixed hash subset already used in October, questions with `sha1(id) % 1013 < 200` (196 questions:
96 open_end, 30 list_recall, 70 number). Each variant is judged once with the official evaluator (gpt-5-mini judge).
Baseline: the re-run's own answers and judgments on the same questions (no new call).

## Phase 1: answer model, open_end subset (96 questions), map unchanged

| variant | answer model (reduce) | reasoning |
|---|---|---|
| S1 | openai/gpt-6.2-astra | effort low |
| S2 | anthropic/claude-opus-5.5 | none |
| S3 | openai/gpt-6.2-sol | effort low |

Winner: the highest judged open_end score of the three; ties go to the cheaper model.

## Phase 2: no relevance map, open_end subset

S4: the Phase 1 winner answers from all 25 pooled items in one call (`--mode single`). Adopted for open_end if it beats
the winner's Phase 1 score on the same 96 questions; otherwise the map stays.

## Phase 3: list_recall and number subsets (30 + 70 questions)

The Phase 1 winner as the reduce model for list and number questions, everything else as in the re-run. Adopted per
type only if it beats the re-run's subset score for that type.

## Phase 4: one full run

The configuration assembled from Phases 1 to 3 runs once on all 1,013 questions and is judged once. Its QS replaces
the row if it is more than 0.3 above 77.85, whatever its rank, and it is reported either way. Nothing is re-run to
improve a draw. If no Phase 1 variant beats the baseline's open_end subset score by at least 3 points, the screening
stops there and the row stays at 77.85.

## Baseline, computed before any call (`harness/screen_2026_11_20.sh baseline`)

The re-run's answers on the subset: open_end 78.12 (96 questions), list_recall 80.71 (30), number 82.86 (70). The
Phase 1 stop rule therefore needs an open_end subset score of at least 81.12.

Driver: `harness/screen_2026_11_20.sh`. `harness/mf_answer_v2.py` gains a third `--mode`, `all`, which sends every
pooled item to the standard reduce call (same prompt, reasoning setting, fallback and post-processing as `mapreduce`);
the two existing modes are unchanged.

# ATM-Bench harness for the "MediaFind + hosted LLMs" leaderboard row

The scripts behind the "MediaFind + hosted LLMs" row on the ATM-Bench leaderboard
(atmbench/atmbench.github.io pull request 6), rebuilt from the October 2026 run logs, and one frozen end-to-end
re-run of them.

- `RECONSTRUCTION.md`: how the harness was rebuilt, how it was checked, and the frozen configuration.
- `REGISTRATION.md`: the conditions written down before the re-run (one run, repeats only for missing or
  incomplete stages, judged once, score reported either way).
- `harness/rerun_frozen.sh`: the end-to-end driver. Every other file under `harness/` is a pipeline step or an
  October experiment script kept as it was.
- `rerun-2026-11-17/RERUN.md`: the re-run's result, pins, timeline and cost; `logs/` holds its logs and
  `SHA256SUMS` the hashes of the release asset.
- Release `rerun-2026-11-17`: the prediction file, the judge output, the per-type answer files, every ranking, the
  rewrite sets and the regenerated item text.

The first-stage ranking is MediaFind's on-device memory search, whose repository is private. The harness calls it
through `harness/mfexp.py` with `MF_WORKTREE` pointing at a MediaFind checkout, and its outputs are published as
rank files so the answer stage can be re-run without it. The answer and memory steps use hosted models through
OpenRouter (`OPENROUTER_API_KEY` read from `~/.config/m3bench/openrouter.env`, or the file named by
`OPENROUTER_ENV_FILE`).

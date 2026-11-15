#!/usr/bin/env bash
# usage: judge.sh <name> <qa.json> [eval_dir_suffix]  — (re)judge ans/<name>.jsonl with gpt-5-mini into a CLEAN eval dir; prints QS
set -euo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"
NAME="$1"; QA="$2"; SUF="${3:-}"
cd "$SP/atm-official"
KEY="$(grep OPENROUTER_API_KEY "${OPENROUTER_ENV_FILE:-$HOME/.config/m3bench/openrouter.env}" | cut -d= -f2- | tr -d "\"' ")"
export OPENAI_API_KEY="$KEY" OPENAI_BASE_URL=https://openrouter.ai/api/v1 PYTHONPATH=.
EV="$SP/ans/$NAME.eval$SUF"; rm -rf "$EV"
"${ATM_PY:-$HOME/eval-archive/atm-run/.venv/bin/python}" memqa/utils/evaluator/evaluate_qa.py --ground-truth "$QA" --predictions "$SP/ans/$NAME.jsonl" \
  --output-dir "$EV" --metrics em atm --judge-provider openai \
  --judge-model openai/gpt-5-mini --judge-reasoning-effort minimal --max-workers "${WORKERS:-6}" > "$SP/ans/$NAME.judge$SUF.log" 2>&1
python3 -c "import json;d=json.load(open('$EV/atm_openai_gpt-5-mini_summary.json'));print('$NAME$SUF QS %.2f'%(100*d['accuracy']), {k:round(100*v['accuracy'],1) for k,v in d['by_qtype'].items()}, 'n', d.get('count'))"

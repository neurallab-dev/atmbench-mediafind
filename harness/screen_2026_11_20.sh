#!/usr/bin/env bash
# Answer-step screening of 2026-11-20 (screening-2026-11-20/PLAN.md) on the fixed sha1 hash subset; one judge pass per variant.
# usage: screen_2026_11_20.sh baseline | phase1 | phase2 MODEL REASONING | phase3 MODEL REASONING
#   REASONING is a JSON reasoning setting for the reduce step, or "" for none.
# Upstream of the answer step everything is the re-run's (rankings in runs/, relevance-map calls replayed from cache/).
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
PY="${ATM_PY:-$HOME/eval-archive/atm-run/.venv/bin/python}"; OUT=runs/screen_2026_11_20.txt
export ITEMS_FILE="$SP/atm/items-san.json" BEST_GUESS=1 FALLBACK_MODEL=anthropic/claude-sonnet-5.5
need() {  # need <estimated $> <stage>: stop when the account's credits (and the key's limit) are below the estimate + $0.5
  local left; left=$("$PY" -I - <<'PY'
import json, os, re, urllib.request
p = os.environ.get("OPENROUTER_ENV_FILE", os.path.expanduser("~/.config/m3bench/openrouter.env"))
k = next(re.match(r"\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(.+)", l).group(1).strip().strip("\"'") for l in open(p) if "OPENROUTER_API_KEY" in l)
get = lambda u: json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"Authorization": "Bearer " + k}), timeout=30))["data"]
key, cr = get("https://openrouter.ai/api/v1/auth/key"), get("https://openrouter.ai/api/v1/credits")
lim = key.get("limit_remaining"); print(min(float(lim) if lim is not None else 1e9, float(cr["total_credits"]) - float(cr["total_usage"])))
PY
)
  echo "    balance \$$left before '$2' (estimate \$$1)" | tee -a "$OUT"
  "$PY" -c "import sys; sys.exit(0 if float('$left') >= $1 + 0.5 else 1)" || { echo "STOP: not enough credits for '$2'" | tee -a "$OUT"; exit 3; }
}
variant() {  # variant NAME TYPE MODEL REASONING MODE ESTIMATE
  local name=$1 t=$2 model=$3 rsn=$4 mode=$5 est=$6 qa="atm/atm-subset-sha1-$2.json" n pool batch prior
  n=$("$PY" -c "import json;print(len(json.load(open('$qa'))))")
  case $t in
    open) pool=runs/san_fused.rank.json; batch=10; prior="";;
    list) pool=runs/san_fused_e_q4b_w15.rank.json; batch=25; prior="RANK_PRIOR_K=0 RANK_PRIOR_K_LIST=10";;
    number) pool=runs/san_fused_e_q4b_w15.rank.json; batch=10; prior="RANK_PRIOR_K=0 RANK_PRIOR_K_LIST=10";;
  esac
  local mapm=deepseek/deepseek-v4.1-flash; [ "$t" = number ] && mapm=anthropic/claude-haiku-5.5
  local script=mf_answer_v2.py; [ "$t" != open ] && script=mf_answer_r2.py
  if [ "$(wc -l < "ans/$name.jsonl" 2>/dev/null || echo 0)" -ne "$n" ] || grep -q '^FAILED' "runs/$name.err" 2>/dev/null; then
    need "$est" "$name"; echo "=== $name: $t, reduce $model ${rsn:-no reasoning}, mode $mode ($(date -u +%FT%TZ))" | tee -a "$OUT"
    env $prior MAP_MODEL=$mapm ANSWER_MODEL="$model" REDUCE_REASONING="$rsn" "$PY" -u $script --qa "$qa" --pool $pool --topk 25 \
      --batch $batch --mode "$mode" --workers 6 --out "ans/$name.jsonl" 2> "runs/$name.err" | tail -1 | tee -a "$OUT"
    echo "    answers $(wc -l < "ans/$name.jsonl"), FAILED $(grep -c '^FAILED' "runs/$name.err"), reduce fallbacks $(grep -c 'reduce fallback' "runs/$name.err"); $(tail -1 "runs/$name.err")" | tee -a "$OUT"
  fi
  grep -q '^FAILED' "runs/$name.err" && { echo "ABORT: $name has FAILED answers" | tee -a "$OUT"; exit 1; }
  [ -s "ans/$name.eval/atm_openai_gpt-5-mini_summary.json" ] || { need 0.3 "judge $name"; WORKERS=6 ./judge.sh "$name" "$SP/$qa" | tee -a "$OUT"; }
}
case "${1:-}" in
  baseline)  # the re-run's own judged answers restricted to the subset: no call
    "$PY" -I - <<'PY' | tee -a "$OUT"
import json
j = {x["id"]: x for x in json.load(open("ans/mediafind-hosted-llms-rerun.eval/atm_openai_gpt-5-mini.json"))}
for t in ("open", "list", "number"):
    ids = [q["id"] for q in json.load(open(f"atm/atm-subset-sha1-{t}.json"))]
    acc = [float(j[i]["accuracy"] in (True, "True")) if j[i].get("metric") == "llm" else float(j[i]["accuracy"]) for i in ids]
    print(f"baseline (re-run of 2026-11-17) {t}: {100 * sum(acc) / len(acc):.2f} on {len(ids)} questions")
PY
    ;;
  phase1)
    variant scr_S1_open open openai/gpt-6.2-astra '{"effort": "low"}' mapreduce 8
    variant scr_S2_open open anthropic/claude-opus-5.5 "" mapreduce 3
    variant scr_S3_open open openai/gpt-6.2-sol '{"effort": "low"}' mapreduce 2 ;;
  phase2) variant scr_S4_open open "$2" "$3" all 16 ;;
  phase3) variant scr_P3_list list "$2" "$3" mapreduce 4; variant scr_P3_number number "$2" "$3" mapreduce 6 ;;
  *) echo "usage: $0 baseline|phase1|phase2 MODEL REASONING|phase3 MODEL REASONING"; exit 2 ;;
esac

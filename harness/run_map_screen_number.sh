#!/usr/bin/env bash
# Map-only relevance screen on the 360 number questions (adopted number policy: old fused pool, 10-item batches) for the given map models.
# Deterministic kept-gold / distractor stats, no reduce. usage: run_map_screen_number.sh MODEL...
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; cd "$SP"
for m in "$@"; do tag="nmap_$(echo "$m" | tr '/.:-' '____')"; echo "=== $m -> $tag ($(date -u +%H:%M) UTC)"
  MAP_MODEL="$m" BASE10=0 QA_FILE="${QA_FILE:-atm/atm-bench-number.json}" POOLK=25 "$HOME/atmbench/.venv/bin/python" -u map_model_test.py "${POOL:-runs/san_fused.rank.json}" "${tag}${POOLTAG:-}" 10 2>&1 | grep -v -i warn
done; echo "=== NUMBER MAP SCREEN DONE ($(date -u +%H:%M) UTC)"

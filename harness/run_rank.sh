#!/usr/bin/env bash
# usage: run_rank.sh <data_dir_name> <tasks.jsonl> <out_prefix> <config>...   (paths relative to this dir; mfexp.py from the worktree with the main venv)
set -euo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"
DATA="$1"; TASKS="$2"; OUT="$3"; shift 3
case "$TASKS" in /*) ;; *) TASKS="$SP/$TASKS";; esac
case "$OUT" in /*) ;; *) OUT="$SP/$OUT";; esac
mkdir -p "$SP/$DATA"
cd "${MF_WORKTREE:-/Users/mbp/code/mediafind/.claude/worktrees/gallant-ptolemy-a222c5}" || { echo "worktree missing: set MF_WORKTREE"; exit 2; }
export HF_HOME="$HOME/Library/Application Support/MediaFind/models/huggingface"
export MEDIAFIND_DATA="$SP/$DATA" MEDIAFIND_EMBED="${MEDIAFIND_EMBED:-BAAI/bge-m3}" MEDIAFIND_RERANK="${MEDIAFIND_RERANK:-BAAI/bge-reranker-v2-m3}"
export MEDIAFIND_MEMORY_ISOLATION=library MEDIAFIND_DEV_MODE=1 MEDIAFIND_PRO=1
exec /Users/mbp/code/mediafind/.venv/bin/python -u "$SP/mfexp.py" "$TASKS" "$OUT" "$@"

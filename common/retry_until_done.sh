#!/usr/bin/env bash
# Re-run one model's NIM job until every prompt has a usable cached answer (or ROUNDS run out).
# Cached answers are reused, so each round only retries the prompts that failed (e.g. during 403 burst blocks).
# Usage: retry_until_done.sh <project_dir> <model> [workers=2] [rounds=12] [pause_s=300]
# The project's src/02_run_models.py must print "done: N ok, M errors"; its rebuild() is called at the end.
PROJECT=$1; MODEL=$2; WORKERS=${3:-2}; ROUNDS=${4:-12}; PAUSE=${5:-300}
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PYTHON:-$ROOT_DIR/.venv/Scripts/python}
[ -x "$PY" ] || PY=$ROOT_DIR/.venv/bin/python
[ -x "$PY" ] || PY=python
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
cd "$PROJECT/src" || exit 1
LOG="../results/logs/run_${MODEL##*/}.txt"
for ((i = 1; i <= ROUNDS; i++)); do
  $PY -u 02_run_models.py "$MODEL" --workers "$WORKERS" >> "$LOG" 2>&1
  LINE=$(grep "done:" "$LOG" | tail -1)
  echo "$(date +%H:%M) ${MODEL##*/} round $i: $LINE"
  [[ "$LINE" == *" 0 errors"* ]] && break
  sleep "$PAUSE"
done
$PY -u -c "import importlib.util as u; s=u.spec_from_file_location('m','02_run_models.py'); m=u.module_from_spec(s); s.loader.exec_module(m); m.rebuild()"
echo "RETRY_DONE ${MODEL##*/} $(date +%H:%M)"

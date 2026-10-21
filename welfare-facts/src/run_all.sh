#!/usr/bin/env bash
# One NIM process per model (2 workers each), then rebuild responses.jsonl. Cached: reruns only redo failures.
cd "$(dirname "$0")" || exit 1
ROOT_DIR="$(cd .. && cd .. && pwd)"  # cwd is src/ after the cd above
PY=${PYTHON:-$ROOT_DIR/.venv/Scripts/python}
[ -x "$PY" ] || PY=$ROOT_DIR/.venv/bin/python
[ -x "$PY" ] || PY=python
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
mkdir -p ../results/logs
MODELS=${MODELS:-"google/gemma-4-31b-it openai/gpt-oss-20b meta/muse-glimmer-30b nvidia/nemotron-3-super-120b-a12b nvidia/nemotron-3.5-lightning-30b-a3b"}
for m in $MODELS; do
  $PY -u 02_run_models.py "$m" --workers 2 > "../results/logs/run_${m##*/}.txt" 2>&1 &
done
wait
$PY -u -c "import importlib.util as u; s=u.spec_from_file_location('m','02_run_models.py'); m=u.module_from_spec(s); s.loader.exec_module(m); m.rebuild()"
echo "FACTS_RUN_DONE $(date +%H:%M)"

#!/usr/bin/env bash
# One NIM process per model variant (4 workers each); cached and resumable. Rebuilds results/responses.jsonl at the end.
cd "$(dirname "$0")" || exit 1
ROOT_DIR="$(cd .. && cd .. && pwd)"  # cwd is src/ after the cd above
PY=${PYTHON:-$ROOT_DIR/.venv/Scripts/python}
[ -x "$PY" ] || PY=$ROOT_DIR/.venv/bin/python
[ -x "$PY" ] || PY=python
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
mkdir -p ../results/logs
$PY -u 02_run_models.py openai/gpt-oss-20b --workers 4 > ../results/logs/run_gpt-oss-20b.txt 2>&1 &
$PY -u 02_run_models.py nvidia/nemotron-3-super-120b-a12b --workers 4 --nothink > ../results/logs/run_nemotron-super-nothink.txt 2>&1 &
$PY -u 02_run_models.py google/gemma-4-31b-it --workers 4 > ../results/logs/run_gemma-4-31b-it.txt 2>&1 &
wait
$PY -u -c "import importlib.util as u; s=u.spec_from_file_location('m','02_run_models.py'); m=u.module_from_spec(s); s.loader.exec_module(m); m.rebuild()"
echo "EXCHANGE_RUN_DONE $(date +%H:%M)"

#!/usr/bin/env bash
# Download each local RM (robust curl fetcher) and score it while the next one downloads.
cd "$(dirname "$0")/../.." || exit 1
ROOT_DIR="$(pwd)"  # cwd is the repo root after the cd above
PY=${PYTHON:-$ROOT_DIR/.venv/Scripts/python}
[ -x "$PY" ] || PY=$ROOT_DIR/.venv/bin/python
[ -x "$PY" ] || PY=python
for pair in "Skywork/Skywork-Reward-V2-Qwen3-0.6B:skywork-v2-qwen3-0.6b" \
            "Skywork/Skywork-Reward-V2-Llama-3.2-1B:skywork-v2-llama3.2-1b" \
            "OpenAssistant/reward-model-deberta-v3-large-v2:oasst-deberta-v3-large" \
            "Skywork/Skywork-Reward-V2-Qwen3-1.7B:skywork-v2-qwen3-1.7b"; do
  repo="${pair%%:*}"; key="${pair##*:}"
  res=$(python common/fetch_curl.py "$repo" | tail -1)
  echo "$(date +%H:%M) $res"
  if echo "$res" | grep -q DOWNLOADED; then
    echo "$(date +%H:%M) SCORING $key"
    (cd reward-model-speciesism/src && HF_HUB_OFFLINE=1 $PY 02_score_local.py "$key" 2>&1 \
       | grep -E "done in|FAILED|loaded|Error" | sed "s/^/$(date +%H:%M) /") &
  fi
done
wait
echo "$(date +%H:%M) ALL_DONE"

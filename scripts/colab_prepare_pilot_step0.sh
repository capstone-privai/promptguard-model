#!/bin/bash
# Rebuild the team's step-0 handoff (promptguard-data ml-train-pilot) in a fresh checkout (e.g. Colab) and
# convert model A and model B to span rows. build_ml_handoff.sh downloads privesc-llm-data at a pinned
# revision and fails unless every file matches the team's committed sha256; prepare_pilot_step0.py checks again.
set -euo pipefail
TEAM=${TEAM_DIR:-/content/promptguard-data-pilot}
[ -d "$TEAM" ] || git clone -q --branch ml-train-pilot https://github.com/capstone-privai/promptguard-data.git "$TEAM"
git -C "$TEAM" checkout -q 5b812c79c204779ed077247c1b5da6d31f9f6ff7
pip install -q datasketch==2.0.0 huggingface_hub
PRIVESC_DIR=${PRIVESC_DIR:-$TEAM/../privesc-llm-data} bash "$TEAM/metric-detection/scripts/build_ml_handoff.sh"
for model in A B; do
  [ -d "runs/data/span-pilot-step0-${model,,}" ] || PYTHONPATH=src python scripts/prepare_pilot_step0.py --team "$TEAM/metric-detection" --model $model
done
python -c "import json; [print('pilot', m, json.load(open(f'runs/data/span-pilot-step0-{m}/manifest.json'))['fingerprint']) for m in 'ab']"

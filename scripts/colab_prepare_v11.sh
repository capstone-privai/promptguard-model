#!/bin/bash
# Rebuild runs/data/span-context-v11 in a fresh checkout (e.g. Colab). The team values cannot be committed,
# but the team build is deterministic, so the rebuilt files must match the original fingerprint.
set -euo pipefail
TEAM=${TEAM_DIR:-/content/promptguard-data}
[ -d "$TEAM" ] || git clone -q https://github.com/capstone-privai/promptguard-data.git "$TEAM"
git -C "$TEAM" checkout -q e1efcd422d6f407c9c2915cd6685445213e661e1
pip install -q datasketch==2.0.0
(cd "$TEAM" \
  && python metric-detection/scripts/ml_train_build.py build metric-detection/templates/ml_train/hand_v1.jsonl --out metric-detection/ml_build \
  && python metric-detection/scripts/ml_train_export.py metric-detection/ml_build/data_test/sessions_ml_hand_v1.jsonl \
       --handoff metric-detection/ml_handoff > /dev/null)
PYTHONPATH=src python scripts/prepare_context_v11.py --team "$TEAM/metric-detection"
python -c "import json; print('v11 fingerprint', json.load(open('runs/data/span-context-v11/manifest.json'))['fingerprint'])"

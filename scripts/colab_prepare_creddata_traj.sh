#!/bin/bash
# Build runs/data/span-creddata-traj-v1 on Colab from CredData-trajectory.zip on Drive (the data is not in Git).
# The files must match the sha256 the local build used, and the split's fingerprint must match the local one.
set -euo pipefail
ZIP=${CREDDATA_TRAJ_ZIP:-/content/drive/MyDrive/promptguard-runs/CredData-trajectory.zip}
OUT=runs/data/span-creddata-traj-v1
[ -d "$OUT" ] && { echo "$OUT exists"; exit 0; }
if [ ! -f "$ZIP" ]; then  # uploaded somewhere else in My Drive: take it from there (the sha256 check below still applies)
  ZIP=$(find /content/drive/MyDrive -maxdepth 4 -name 'CredData-trajectory*.zip' 2>/dev/null | head -1)
fi
test -n "$ZIP" && test -f "$ZIP" || { echo "CredData-trajectory.zip not found in My Drive; upload it to promptguard-runs" >&2; exit 1; }
echo "data zip: $ZIP"
SRC=/content/creddata-traj
rm -rf "$SRC" && mkdir -p "$SRC" && unzip -q "$ZIP" -d "$SRC" -x '__MACOSX/*'
python - "$SRC/CredData-trajectory" <<'EOF'
import hashlib, sys
from pathlib import Path
expected = {
    'sessions_train.jsonl': '2a3e3756cc45ffaf541e9198ee08262553ef94d5329bb4d5dce91079ac9a3a18',
    'gold_train.jsonl': '0a120690e50c42a56b075b52452adc542252a29be110a5f9f7a941aa9b5e180d',
    'labels_train.jsonl': '1c15924afda5ace3b8b13df19dcab51d8b21ac1aaba743398e739476d4d1e167',
    'manifest.json': '150dc684f54eb66186bf8b9ebe3a325b2cb0cf4ccba8b0f8e8998abbfa8e8fe2',
}
for name, sha in expected.items():
    if hashlib.sha256((Path(sys.argv[1]) / name).read_bytes()).hexdigest() != sha:
        sys.exit(f'{name} differs from the file the local split was built from')
print('CredData-trajectory files OK')
EOF
PYTHONPATH=src python scripts/prepare_creddata_traj.py --src "$SRC/CredData-trajectory" > /dev/null
python -c "import json, sys; f = json.load(open('$OUT/manifest.json'))['fingerprint']; print('split fingerprint', f); sys.exit(0 if f == '19c26b1e60c435f5b16afe702ea860cc5aa712811cf0564a6c1712c9d94fb40c' else 'split differs from the local one')"

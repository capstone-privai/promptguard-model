#!/bin/bash
# Build runs/data/span-traj-combined-v1 (Trajectory-train-combined v1, the team's third training set) from its zip
# (the data is not in Git): on Colab from Drive, on a local PC from TRAJ_COMBINED_ZIP.
# The earlier CredData + NoseyParker split is built first (scripts/colab_prepare_creddata_np_traj.sh, which checks its
# own sha256 and fingerprint); its groups keep their split. The zip's files must match the sha256 the local build used,
# and the split's fingerprint must match the local one.
set -euo pipefail
OUT=runs/data/span-traj-combined-v1
[ -d "$OUT" ] && { echo "$OUT exists"; exit 0; }
bash scripts/colab_prepare_creddata_np_traj.sh
ZIP=${TRAJ_COMBINED_ZIP:-/content/drive/MyDrive/promptguard-runs/Trajectory-train-combined.zip}
if [ ! -f "$ZIP" ] && [ -d /content/drive/MyDrive ]; then  # elsewhere in My Drive: take it from there (sha256 still checked)
  ZIP=$(find /content/drive/MyDrive -maxdepth 4 -name 'Trajectory-train-combined*.zip' 2>/dev/null | head -1)
fi
test -n "$ZIP" && test -f "$ZIP" || { echo "Trajectory-train-combined.zip not found: upload it to Drive promptguard-runs, or set TRAJ_COMBINED_ZIP" >&2; exit 1; }
echo "data zip: $ZIP"
if [ -d /content ]; then SRC=${TC_UNZIP_DIR:-/content/traj-combined}; else SRC=${TC_UNZIP_DIR:-runs/traj-combined-src}; fi
rm -rf "$SRC" && mkdir -p "$SRC" && unzip -q "$ZIP" -d "$SRC" -x '__MACOSX/*'
python - "$SRC/Trajectory-train-combined" <<'EOF'
import hashlib, sys
from pathlib import Path
expected = {
    'sessions_train.jsonl': '6fbb276791f993fcb97d79f0fe553fc3a109b5aefca9caa7b5220549c8fbe4e7',
    'gold_train.jsonl': '1a0a9c1e1f0ecd9612680cf14f5c43347453611674b583233e930a282bf1b947',
    'labels_train.jsonl': 'a74312d08cc041a47e5ac5b993165b19fba89e22b35589dcb99ec62d89a43c8e',
    'manifest.json': '384685a4f8adc66863e48108e14cdc0fd63f6cd141397c533c02d7b837790b9a',
    'README.md': '4579abe99962713f770e3a740c6661febd99afd698df1a8e2510840c82d2070c',
}
for name, sha in expected.items():
    if hashlib.sha256((Path(sys.argv[1]) / name).read_bytes()).hexdigest() != sha:
        sys.exit(f'{name} differs from the file the local split was built from')
print('Trajectory-train-combined files OK')
EOF
PYTHONPATH=src python scripts/prepare_traj_combined.py --src "$SRC/Trajectory-train-combined" > /dev/null
python -c "import json, sys; f = json.load(open('$OUT/manifest.json'))['fingerprint']; print('split fingerprint', f); sys.exit(0 if f == '137967461711ccd1a23b3351001b8c308af6db790b3b2c451c6953714cce91cf' else 'split differs from the local one')"

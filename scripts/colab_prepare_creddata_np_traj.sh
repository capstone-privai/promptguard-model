#!/bin/bash
# Build runs/data/span-creddata-np-traj-v1 (CredData-trajectory v1 plus NoseyParker-trajectory v1 train) from the two
# zips (the data is not in Git): on Colab from Drive, on a local PC from CREDDATA_TRAJ_ZIP and NP_TRAJ_ZIP.
# The CredData split is built first (scripts/colab_prepare_creddata_traj.sh, which checks its own sha256 and
# fingerprint). The NP files must match the sha256 the local build used, and the combined split's fingerprint must
# match the local one.
set -euo pipefail
OUT=runs/data/span-creddata-np-traj-v1
[ -d "$OUT" ] && { echo "$OUT exists"; exit 0; }
bash scripts/colab_prepare_creddata_traj.sh
ZIP=${NP_TRAJ_ZIP:-/content/drive/MyDrive/promptguard-runs/NoseyParker-trajectory.zip}
if [ ! -f "$ZIP" ] && [ -d /content/drive/MyDrive ]; then  # elsewhere in My Drive: take it from there (sha256 still checked)
  ZIP=$(find /content/drive/MyDrive -maxdepth 4 -name 'NoseyParker-trajectory*.zip' 2>/dev/null | head -1)
fi
test -n "$ZIP" && test -f "$ZIP" || { echo "NoseyParker-trajectory.zip not found: upload it to Drive promptguard-runs, or set NP_TRAJ_ZIP" >&2; exit 1; }
echo "data zip: $ZIP"
if [ -d /content ]; then SRC=${NP_UNZIP_DIR:-/content/noseyparker-traj}; else SRC=${NP_UNZIP_DIR:-runs/noseyparker-traj-src}; fi
rm -rf "$SRC" && mkdir -p "$SRC" && unzip -q "$ZIP" -d "$SRC" -x '__MACOSX/*'
python - "$SRC/NoseyParker-trajectory" <<'EOF'
import hashlib, sys
from pathlib import Path
expected = {
    'sessions_train.jsonl': '843d9a01275b9082afc215d25b33478be8de38bb0c13c61c05949b80470de781',
    'gold_train.jsonl': 'a354b568e932024a1f2631891f8d795142c09f3d92f9ce2e20f17855ebebe16e',
    'labels_train.jsonl': '54e8c0c1904a6a2f13567d17b54b11a4b4004cf2bf42c1d3c9a8c2e45226d6e5',
    'manifest.json': 'b7735d4d1c66546fce76c99f6605de5de924884c20f17ce61ba3a2360690b6b7',
}
for name, sha in expected.items():
    if hashlib.sha256((Path(sys.argv[1]) / name).read_bytes()).hexdigest() != sha:
        sys.exit(f'{name} differs from the file the local split was built from')
print('NoseyParker-trajectory files OK')
EOF
PYTHONPATH=src python scripts/prepare_creddata_np_traj.py --np-src "$SRC/NoseyParker-trajectory" > /dev/null
python -c "import json, sys; f = json.load(open('$OUT/manifest.json'))['fingerprint']; print('split fingerprint', f); sys.exit(0 if f == 'c7d0de5dc012c8040e17cfa2027956090c8dd90dda56d3caea4746e9ec0c8b8e' else 'split differs from the local one')"

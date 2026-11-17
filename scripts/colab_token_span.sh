#!/bin/bash
# Train a token-classification span detector on CredData-trajectory v1 on a Colab GPU, or score the frozen ones on test.
#
#   bash scripts/colab_token_span.sh mmbert|qwen|qwen-bidir   # train; epoch and cutoff are chosen on validation only
#   bash scripts/colab_token_span.sh test [model ...]         # after selection: score each frozen model on test once
#                                                             # (default: mmbert qwen)
#
# Run from the repository folder after Drive is mounted and the package is installed. Prints counts only.
# Qwen writes everything (LoRA adapters ~50 MB per epoch, a ~0.1 GB resumable checkpoint) to Drive and resumes
# after a runtime reset with the same command. mmBERT trains on the runtime's disk (a full checkpoint is ~1.7 GB),
# restarts from scratch after a reset (an epoch is well under a runtime), and copies only the selected epoch to Drive.
set -euo pipefail
m=$1
DRIVE=${DRIVE_DIR:-/content/drive/MyDrive/promptguard-runs}
export PYTHONPATH=src USE_TF=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no GPU: stop here, do not train on CPU')"
python -c "import peft" 2>/dev/null || pip install -q "peft>=0.13,<1"
pip uninstall -q -y torchao 2>/dev/null || true  # peft rejects Colab's torchao, which is only used for quantization
bash scripts/colab_prepare_creddata_traj.sh
free_gb() { python -c "import shutil; print(round(shutil.disk_usage('$DRIVE').free / 2**30, 2))"; }
need() { python -c "import sys; sys.exit(0 if $(free_gb) >= $1 else 'Drive has under $1 GB free: stop')"; }

case $m in
  qwen|qwen-bidir)
    OUT=$DRIVE/token-span-creddata-$m
    need 0.6
    resume=(); [ -d "$OUT" ] && resume=(--resume)  # starts fresh if no checkpoint was saved yet
    python scripts/token_span.py train --config configs/token-span-creddata-$m.yaml --set output_dir=$OUT "${resume[@]}" 2>&1 \
      | grep --line-buffered -v -i warn | tee -a "$OUT.log"
    test -f "$OUT/selection.json"
    ;;
  mmbert)
    LOCAL=/content/runs/token-span/creddata-mmbert
    OUT=$DRIVE/token-span-creddata-mmbert
    need 0.7
    rm -rf "$LOCAL"
    python scripts/token_span.py train --config configs/token-span-creddata-mmbert.yaml --set output_dir=$LOCAL 2>&1 \
      | grep --line-buffered -v -i warn | tee -a "$DRIVE/token-span-creddata-mmbert.log"
    e=$(python -c "import json; print(json.load(open('$LOCAL/selection.json'))['selected']['epoch'])")
    mkdir -p "$OUT"
    cp "$LOCAL"/{selection.json,history.json,run.json,config.json,progress.jsonl} "$OUT/"
    cp -r "$LOCAL/epoch-$e" "$OUT/"
    ;;
  test)
    models=("${@:2}"); [ ${#models[@]} -gt 0 ] || models=(mmbert qwen)
    for model in "${models[@]}"; do
      OUT=$DRIVE/token-span-creddata-$model
      read e t < <(python -c "import json; s = json.load(open('$OUT/selection.json'))['selected']; print(s['epoch'], repr(s['threshold']))")
      python scripts/token_span.py test --config configs/token-span-creddata-$model.yaml --weights "$OUT/epoch-$e" \
        --threshold "$t" --split test 2>/dev/null | tail -1 | tee "$OUT/test.json"
    done
    exit 0
    ;;
  *) echo "usage: $0 mmbert|qwen|qwen-bidir|test [model ...]" >&2; exit 2 ;;
esac
python -c "import json, sys; s = json.load(open(sys.argv[1])); print(json.dumps(dict(selected=s['selected'], history=[dict(epoch=h['epoch'], loss=round(h['loss'], 5), raw={k: h['raw'][k] for k in ('tp', 'fp', 'fn', 'f2')}, selected={k: h['selected'][k] for k in ('threshold', 'tp', 'fp', 'fn', 'precision', 'f2')}) for h in s['history']]), separators=(',', ':')))" "$OUT/selection.json"

#!/bin/bash
# Back-end verifier on CredData-trajectory v1 (Colab, or a local PC with bash):
#
#   bash scripts/colab_verifier.sh prepare               # detector candidates + false-positive analysis (CPU is fine)
#   bash scripts/colab_verifier.sh train MODEL           # MODEL: configs/verifier-creddata-MODEL.yaml
#                                                        #   mmbert-small, mmbert-base     local stage, full fine-tune (T4)
#                                                        #   qwen1.5b-bidir                server stage, LoRA (T4)
#                                                        #   qwen7b-bidir                  server stage, LoRA, bfloat16 (L4 or A100)
#                                                        #   qwen14b-bidir                 server stage, LoRA, bfloat16 (A100 40 GB+)
#   bash scripts/colab_verifier.sh system MODEL          # validation: union vs strong-only vs strong+verifier
#   bash scripts/colab_verifier.sh test MODEL            # once, after the run is frozen: candidate and system level
#
# DRIVE_DIR (default /content/drive/MyDrive/promptguard-runs) keeps results; DETECTOR_DIR (default
# ../promptguard-claude-demoV0) is the PromptGuard checkout whose detector makes the candidates (cloned if missing).
# The candidate data holds values: it stays under runs/ and in DRIVE_DIR, like the source zip, and is never committed.
# mmBERT trains in LOCAL_RUNS and is copied to DRIVE_DIR at the end. Qwen writes to DRIVE_DIR as it goes (LoRA adapter,
# a resumable checkpoint every 30 minutes) and the same train command resumes it after a runtime reset.
set -euo pipefail
cmd=${1:?usage: $0 prepare | train MODEL | system MODEL | test MODEL}
m=${2:-}
DRIVE=${DRIVE_DIR:-/content/drive/MyDrive/promptguard-runs}
LOCAL_ROOT=${LOCAL_RUNS:-runs}
DETECTOR=${DETECTOR_DIR:-../promptguard-claude-demoV0}
DATA=runs/data/verifier-creddata-traj-v1
SAVED=$DRIVE/data-verifier-creddata-traj-v1
export PYTHONPATH=src USE_TF=0
mkdir -p "$DRIVE"
threshold() { python -c "import json; b = json.load(open('$1/model/bundle.json')); print(b['thresholds'][str(b['deployment_recall_target'])])"; }

case $cmd in
  prepare)
    if [ -d "$DATA" ]; then echo "$DATA exists"
    elif [ -d "$SAVED" ]; then mkdir -p runs/data && cp -r "$SAVED" "$DATA" && echo "restored $DATA from $SAVED"
    else
      bash scripts/colab_prepare_creddata_traj.sh
      [ -d "$DETECTOR" ] || git clone --depth 1 https://github.com/capstone-privai/promptguard-claude-demoV0 "$DETECTOR"
      python -c "import credsweeper" 2>/dev/null || pip install -q -r "$DETECTOR/requirements.txt"
      python scripts/prepare_verifier_candidates.py --detector "$DETECTOR" --data runs/data/span-creddata-traj-v1 --out "$DATA"
      cp -r "$DATA" "$SAVED"
    fi
    python -c "import json; a = json.load(open('$DATA/analysis.json')); print(json.dumps({s: dict(gold=v['gold_spans'], candidates=v['candidates'], by_tier=v['by_tier'], gold_reached=v['gold_reached']) for s, v in a.items()}, ensure_ascii=False))"
    ;;
  train)
    test -f "configs/verifier-creddata-$m.yaml" || { echo "no configs/verifier-creddata-$m.yaml" >&2; exit 2; }
    test -d "$DATA" || { echo "run: bash $0 prepare" >&2; exit 2; }
    OUT=$DRIVE/verifier-creddata-$m
    case $m in
      qwen*)
        python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no GPU: stop here, do not train on CPU')"
        # The marker rows are trained with PEFT trainable tokens (peft 0.15 or later).
        python -c "import peft, sys; sys.exit(tuple(map(int, peft.__version__.split('.')[:2])) < (0, 15))" 2>/dev/null \
          || pip install -q "peft>=0.15,<1"
        if [ -d /content ]; then pip uninstall -q -y torchao 2>/dev/null || true; fi  # peft rejects Colab's torchao
        [ -f "$OUT/model/bundle.json" ] && { echo "$OUT is finished"; exit 0; }
        resume=(); [ -d "$OUT" ] && resume=(--resume)  # starts fresh if no checkpoint was saved yet
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train.py --config "configs/verifier-creddata-$m.yaml" \
          --set output_dir="$OUT" "${resume[@]}" 2>&1 | grep --line-buffered -v -i warn | tee -a "$OUT.log"
        test -f "$OUT/model/bundle.json"
        ;;
      *)
        python -c "import torch; print('GPU' if torch.cuda.is_available() else 'no GPU: training on CPU (slow)')"
        LOCAL=$LOCAL_ROOT/verifier/creddata-$m
        [ -e "$OUT" ] && { echo "$OUT exists: the trainer never overwrites a run" >&2; exit 2; }
        rm -rf "$LOCAL"
        python scripts/train.py --config "configs/verifier-creddata-$m.yaml" --set output_dir="$(realpath -m "$LOCAL")" 2>&1 \
          | grep --line-buffered -v -i warn | tee "$DRIVE/verifier-creddata-$m.log"
        cp -r "$LOCAL" "$OUT"
        ;;
    esac
    ;;
  system)
    OUT=$DRIVE/verifier-creddata-$m
    python scripts/evaluate_verifier_system.py --data "$DATA" --predictions "$OUT/validation_predictions.jsonl" \
      --threshold "$(threshold "$OUT")" --split validation | tee "$OUT/system_validation.json"
    ;;
  test)
    OUT=$DRIVE/verifier-creddata-$m
    [ -e "$OUT/final" ] && { echo "$OUT/final exists: test was already scored" >&2; exit 2; }
    python scripts/final_evaluate.py --data "$DATA/ambiguous/candidates.jsonl.gz" --run "$OUT" --output "$OUT/final"
    python scripts/evaluate_verifier_system.py --data "$DATA" --predictions "$OUT/final/test_predictions.jsonl" \
      --threshold "$(threshold "$OUT")" --split test | tee "$OUT/system_test.json"
    ;;
  *) echo "usage: $0 prepare | train MODEL | system MODEL | test MODEL" >&2; exit 2 ;;
esac

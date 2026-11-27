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
# CredData-trajectory plus NoseyParker-trajectory train (validation only; `test` refuses, the NP test is the team's):
#   bash scripts/colab_verifier.sh prepare np            # the same detector as the CredData build, pinned: commit
#                                                        # 668c2c1, CredSweeper 1.19.0, gitleaks 8.30.1; stops unless
#                                                        # the CredData candidates come out as in that build
#   bash scripts/colab_verifier.sh train np-mmbert-small # (or np-mmbert-base), then: system np-mmbert-small
#
# Trajectory-train-combined v1, the third training set (validation only; `test` refuses, as for NP):
#   bash scripts/colab_verifier.sh prepare tc            # the same pinned detector on runs/data/span-traj-combined-v1
#   bash scripts/colab_verifier.sh train tc-mmbert-small # configs/verifier-traj-combined-mmbert-small.yaml, then:
#                                                        # system tc-mmbert-small (results in verifier-traj-combined-...)
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
NAME=creddata-$m  # configs/verifier-$NAME.yaml, results in $DRIVE/verifier-$NAME
case $m in
  np|np-*)
    DATA=runs/data/verifier-creddata-np-traj-v1
    SAVED=$DRIVE/data-verifier-creddata-np-traj-v1 ;;
  tc|tc-*)
    DATA=runs/data/verifier-traj-combined-v1
    SAVED=$DRIVE/data-verifier-traj-combined-v1
    NAME=traj-combined-${m#tc-} ;;
esac
DETECTOR_COMMIT=668c2c1e107aff596c07f3dd5e520ba76b0fbdf0  # the detector the CredData-trajectory candidates came from
GITLEAKS_VERSION=8.30.1
export PYTHONPATH=src USE_TF=0
mkdir -p "$DRIVE"
threshold() { python -c "import json; b = json.load(open('$1/model/bundle.json')); print(b['thresholds'][str(b['deployment_recall_target'])])"; }

case $cmd in
  prepare)
    if [ -d "$DATA" ]; then echo "$DATA exists"
    elif [ -d "$SAVED" ]; then mkdir -p runs/data && cp -r "$SAVED" "$DATA" && echo "restored $DATA from $SAVED"
    elif [ "$m" = np ] || [ "$m" = tc ]; then
      if [ "$m" = np ]; then
        bash scripts/colab_prepare_creddata_np_traj.sh; SPAN=runs/data/span-creddata-np-traj-v1
      else
        bash scripts/colab_prepare_traj_combined.sh; SPAN=runs/data/span-traj-combined-v1
      fi
      DET=${DETECTOR_DIR:-runs/detector-668c2c1}
      if [ ! -d "$DET" ]; then
        git clone -q https://github.com/capstone-privai/promptguard-claude-demoV0 "$DET"
        git -C "$DET" checkout -q "$DETECTOR_COMMIT"
      fi
      [ "$(git -C "$DET" rev-parse HEAD)" = "$DETECTOR_COMMIT" ] || { echo "$DET is not at $DETECTOR_COMMIT" >&2; exit 2; }
      python -c "import credsweeper, sys; sys.exit(credsweeper.__version__ != '1.19.0')" 2>/dev/null || pip install -q credsweeper==1.19.0
      if ! gitleaks version 2>/dev/null | grep -qxE "v?$GITLEAKS_VERSION"; then  # the official release, checked against its checksums
        case "$(uname -s)-$(uname -m)" in
          Linux-x86_64) asset=linux_x64 ;; Darwin-arm64) asset=darwin_arm64 ;; Darwin-x86_64) asset=darwin_x64 ;;
          *) echo "no gitleaks $GITLEAKS_VERSION build for this machine" >&2; exit 2 ;;
        esac
        tgz=gitleaks_${GITLEAKS_VERSION}_$asset.tar.gz
        url=https://github.com/gitleaks/gitleaks/releases/download/v$GITLEAKS_VERSION
        mkdir -p runs/bin/gitleaks-dl && (cd runs/bin/gitleaks-dl \
          && curl -fsSLO "$url/$tgz" && curl -fsSLO "$url/gitleaks_${GITLEAKS_VERSION}_checksums.txt" \
          && grep " $tgz\$" "gitleaks_${GITLEAKS_VERSION}_checksums.txt" | { command -v sha256sum >/dev/null && sha256sum -c - || shasum -a 256 -c -; } \
          && tar -xzf "$tgz" gitleaks && mv gitleaks ..)
        export PATH=$PWD/runs/bin:$PATH
      fi
      gitleaks version
      python scripts/prepare_verifier_candidates.py --detector "$DET" --data "$SPAN" --out "$DATA"
      # NP: the CredData rows are unchanged, so their train and validation candidates must be the earlier build's.
      # (tc relabels the CredData rows, so their candidates' labels may differ; the detector is the same pinned one.)
      [ "$m" = tc ] || python - "$DATA" <<'EOF'
import gzip, json, sys
from collections import Counter
seen = Counter()
with gzip.open(f'{sys.argv[1]}/all/candidates.jsonl.gz', 'rt', encoding='utf-8') as f:
    for r in map(json.loads, f):
        if r['source']['row_id'].startswith('creddata_traj:') and r['source']['split'] != 'test':
            seen[f"{r['source']['split']}/{r['detector']['strength']}/{r['label']['action']}"] += 1
earlier = {'train/strong/MASK': 5911, 'train/strong/KEEP': 2174, 'train/ambiguous/MASK': 2157, 'train/ambiguous/KEEP': 2750,
           'validation/strong/MASK': 601, 'validation/strong/KEEP': 456, 'validation/ambiguous/MASK': 395,
           'validation/ambiguous/KEEP': 809}
print('CredData candidates', dict(seen))
sys.exit(0 if dict(seen) == earlier else 'CredData candidates differ from the earlier build: the detector setup differs, stop')
EOF
      cp -r "$DATA" "$SAVED"
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
    test -f "configs/verifier-$NAME.yaml" || { echo "no configs/verifier-$NAME.yaml" >&2; exit 2; }
    test -d "$DATA" || { echo "run: bash $0 prepare" >&2; exit 2; }
    OUT=$DRIVE/verifier-$NAME
    case $m in
      qwen*)
        python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no GPU: stop here, do not train on CPU')"
        # The marker rows are trained with PEFT trainable tokens (peft 0.15 or later).
        python -c "import peft, sys; sys.exit(tuple(map(int, peft.__version__.split('.')[:2])) < (0, 15))" 2>/dev/null \
          || pip install -q "peft>=0.15,<1"
        if [ -d /content ]; then pip uninstall -q -y torchao 2>/dev/null || true; fi  # peft rejects Colab's torchao
        [ -f "$OUT/model/bundle.json" ] && { echo "$OUT is finished"; exit 0; }
        resume=(); [ -d "$OUT" ] && resume=(--resume)  # starts fresh if no checkpoint was saved yet
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train.py --config "configs/verifier-$NAME.yaml" \
          --set output_dir="$OUT" "${resume[@]}" 2>&1 | grep --line-buffered -v -i warn | tee -a "$OUT.log"
        test -f "$OUT/model/bundle.json"
        ;;
      *)
        python -c "import torch; print('GPU' if torch.cuda.is_available() else 'no GPU: training on CPU (slow)')"
        LOCAL=$LOCAL_ROOT/verifier/$NAME
        [ -e "$OUT" ] && { echo "$OUT exists: the trainer never overwrites a run" >&2; exit 2; }
        rm -rf "$LOCAL"
        python scripts/train.py --config "configs/verifier-$NAME.yaml" --set output_dir="$(realpath -m "$LOCAL")" 2>&1 \
          | grep --line-buffered -v -i warn | tee "$DRIVE/verifier-$NAME.log"
        cp -r "$LOCAL" "$OUT"
        ;;
    esac
    ;;
  system)
    OUT=$DRIVE/verifier-$NAME
    python scripts/evaluate_verifier_system.py --data "$DATA" --predictions "$OUT/validation_predictions.jsonl" \
      --threshold "$(threshold "$OUT")" --split validation | tee "$OUT/system_validation.json"
    ;;
  test)
    case $m in np-*|tc-*) echo "$m: NP models are scored on validation only; the NP test is the team's hold-out" >&2; exit 2 ;; esac
    OUT=$DRIVE/verifier-$NAME
    [ -e "$OUT/final" ] && { echo "$OUT/final exists: test was already scored" >&2; exit 2; }
    python scripts/final_evaluate.py --data "$DATA/ambiguous/candidates.jsonl.gz" --run "$OUT" --output "$OUT/final"
    python scripts/evaluate_verifier_system.py --data "$DATA" --predictions "$OUT/final/test_predictions.jsonl" \
      --threshold "$(threshold "$OUT")" --split test | tee "$OUT/system_test.json"
    ;;
  *) echo "usage: $0 prepare | train MODEL | system MODEL | test MODEL" >&2; exit 2 ;;
esac

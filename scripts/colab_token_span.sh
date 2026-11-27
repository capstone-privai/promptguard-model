#!/bin/bash
# Train a token-classification span detector on CredData-trajectory v1 on a GPU (Colab or a local PC), or score the
# frozen ones on test.
#
#   bash scripts/colab_token_span.sh MODEL [--set key=value ...]   # train; epoch and cutoff chosen on validation only
#   bash scripts/colab_token_span.sh test [MODEL ...]               # after selection: score each frozen model on test once
#                                                                   # (default: mmbert qwen)
#
# MODEL is a config name, configs/token-span-creddata-MODEL.yaml:
#   mmbert, mmbert-base                                   full fine-tune (T4 is enough)
#   qwen, qwen-bidir, qwen1.5b-bidir                      LoRA, float16 autocast (T4 is enough)
#   qwen7b-bidir                                          LoRA, bfloat16 (L4 24 GB or A100)
#   qwen14b-bidir                                         LoRA, bfloat16 (A100 40 GB or larger)
#   qwen14b-bidir --set load_in_4bit=true                 QLoRA on a 24 GB GPU; results go to ...-4bit, test it as
#                                                         `test qwen14b-bidir-4bit`
#   np-mmbert, np-mmbert-base, np-qwen-bidir, np-qwen1.5b-bidir
#                                                         the same models on CredData-trajectory plus NoseyParker-
#                                                         trajectory train (also needs NP_TRAJ_ZIP or the zip on
#                                                         Drive). Validation only: `test` refuses them, since the NP
#                                                         test is the team's hold-out.
#   tc-mmbert-base                                        mmBERT-base on Trajectory-train-combined v1, the third
#                                                         training set (configs/token-span-traj-combined-mmbert-base.yaml,
#                                                         results in token-span-traj-combined-mmbert-base; needs the zip
#                                                         on Drive or TRAJ_COMBINED_ZIP). Validation only, as for NP.
#
# Run from the repository folder after the package is installed. Prints counts only.
# Colab: mount Drive first; results go to Drive (DRIVE_DIR, default /content/drive/MyDrive/promptguard-runs).
# Local PC: set DRIVE_DIR to a results folder, CREDDATA_TRAJ_ZIP to the data zip, and LOCAL_RUNS to a scratch
# folder, e.g.  DRIVE_DIR=~/promptguard-runs CREDDATA_TRAJ_ZIP=~/CredData-trajectory.zip LOCAL_RUNS=runs/local \
#               bash scripts/colab_token_span.sh mmbert-base
# Qwen writes everything (LoRA adapters per epoch, a resumable checkpoint) to DRIVE_DIR and resumes after a reset
# with the same command. mmBERT trains in LOCAL_RUNS (a full checkpoint is ~1.7 GB small, ~3.7 GB base), restarts
# from scratch after a reset, and copies only the selected epoch to DRIVE_DIR.
set -euo pipefail
m=${1:?usage: $0 MODEL [--set key=value ...] | test [MODEL ...]}
DRIVE=${DRIVE_DIR:-/content/drive/MyDrive/promptguard-runs}
LOCAL_ROOT=${LOCAL_RUNS:-/content/runs}
export PYTHONPATH=src USE_TF=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$DRIVE"
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no GPU: stop here, do not train on CPU')"
python -c "import peft" 2>/dev/null || pip install -q "peft>=0.13,<1"
if [ -d /content ]; then pip uninstall -q -y torchao 2>/dev/null || true; fi  # peft rejects Colab's torchao (quantization only)
bash scripts/colab_prepare_creddata_traj.sh
case $m in np-*) bash scripts/colab_prepare_creddata_np_traj.sh ;; tc-*) bash scripts/colab_prepare_traj_combined.sh ;; esac
free_gb() { python -c "import shutil; print(round(shutil.disk_usage('$DRIVE').free / 2**30, 2))"; }
need() { python -c "import sys; sys.exit(0 if $(free_gb) >= $1 else 'Drive has under $1 GB free: stop')"; }
run_name() { case $1 in tc-*) echo "traj-combined-${1#tc-}" ;; *) echo "creddata-$1" ;; esac; }
config_of() { echo "configs/token-span-$(run_name "${1%-4bit}").yaml"; }
sets_of() { case $1 in *-4bit) echo "--set load_in_4bit=true" ;; esac; }

case $m in
  test)
    models=("${@:2}"); [ ${#models[@]} -gt 0 ] || models=(mmbert qwen)
    for model in "${models[@]}"; do
      case $model in np-*|tc-*) echo "$model: NP models are scored on validation only; the NP test is the team's hold-out" >&2; exit 2 ;; esac
      OUT=$DRIVE/token-span-creddata-$model
      read e t < <(python -c "import json; s = json.load(open('$OUT/selection.json'))['selected']; print(s['epoch'], repr(s['threshold']))")
      # shellcheck disable=SC2046
      python scripts/token_span.py test --config "$(config_of "$model")" $(sets_of "$model") --weights "$OUT/epoch-$e" \
        --threshold "$t" --split test 2>/dev/null | tail -1 | tee "$OUT/test.json"
    done
    exit 0
    ;;
  qwen*|np-qwen*)
    CONFIG=$(config_of "$m"); test -f "$CONFIG" || { echo "no $CONFIG" >&2; exit 2; }
    extra=("${@:2}")
    name=$m
    if [ ${#extra[@]} -gt 0 ] && printf '%s\n' "${extra[@]}" | grep -q 'load_in_4bit=true'; then
      name=$m-4bit
      python -c "import bitsandbytes" 2>/dev/null || pip install -q "bitsandbytes>=0.43"
    fi
    OUT=$DRIVE/token-span-creddata-$name
    case $m in qwen14b*) need 2.5 ;; qwen7b*) need 1.5 ;; *) need 0.6 ;; esac
    resume=(); [ -d "$OUT" ] && resume=(--resume)  # starts fresh if no checkpoint was saved yet
    python scripts/token_span.py train --config "$CONFIG" --set output_dir="$OUT" "${extra[@]}" "${resume[@]}" 2>&1 \
      | grep --line-buffered -v -i warn | tee -a "$OUT.log"
    test -f "$OUT/selection.json"
    ;;
  mmbert*|np-mmbert*|tc-mmbert*)
    CONFIG=$(config_of "$m"); test -f "$CONFIG" || { echo "no $CONFIG" >&2; exit 2; }
    LOCAL=$LOCAL_ROOT/token-span/$(run_name "$m")
    OUT=$DRIVE/token-span-$(run_name "$m")
    case $m in
      tc-*)
        # Resumable across runtimes: Colab reclaims a runtime after about 8 hours, and 4 epochs of mmBERT-base on the
        # third training set take about 9.5 hours on a T4. The checkpoint (~3.7 GB) and the best epoch so far
        # (~1.2 GB) sit on Drive; the same command resumes from the last finished epoch on a new runtime.
        resume=()
        if [ -f "$OUT/last/state.json" ]; then resume=(--resume); else need 4.8; rm -rf "$LOCAL" "$OUT/best"; fi
        mkdir -p "$OUT"
        python scripts/token_span.py train --config "$CONFIG" --set output_dir="$LOCAL" --set checkpoint_dir="$OUT/last" \
          --set checkpoint_staging="$LOCAL_ROOT/ckpt-staging/$(run_name "$m")" --set best_dir="$OUT/best" \
          "${resume[@]}" "${@:2}" 2>&1 | grep --line-buffered -v -i warn | tee -a "$DRIVE/token-span-$(run_name "$m").log"
        e=$(python -c "import json; print(json.load(open('$LOCAL/selection.json'))['selected']['epoch'])")
        cp "$LOCAL"/{history.json,run.json,config.json,progress.jsonl} "$OUT/"
        rm -rf "$OUT/last"
        if [ -d "$LOCAL/epoch-$e" ]; then
          # Trained on this runtime: it stays in $LOCAL (Drive has no room for another copy, since a deleted file
          # stays in the Drive trash and still counts); colab_tc_all.sh publishes the bundle from $LOCAL.
          rm -rf "$OUT/best"
        else
          b=$(python -c "import json; print(json.load(open('$OUT/best/epoch.json'))['epoch'])")
          [ "$b" = "$e" ] || { echo "best epoch on Drive is $b, selection chose $e" >&2; exit 1; }
          rm -rf "$OUT/epoch-$e" && mv "$OUT/best" "$OUT/epoch-$e" && rm "$OUT/epoch-$e/epoch.json"
        fi
        cp "$LOCAL/selection.json" "$OUT/"  # last: selection.json marks the step finished
        ;;
      *)
        case $m in *mmbert-base) need 1.5 ;; *) need 0.7 ;; esac
        rm -rf "$LOCAL"
        python scripts/token_span.py train --config "$CONFIG" --set output_dir="$LOCAL" "${@:2}" 2>&1 \
          | grep --line-buffered -v -i warn | tee -a "$DRIVE/token-span-$(run_name "$m").log"
        e=$(python -c "import json; print(json.load(open('$LOCAL/selection.json'))['selected']['epoch'])")
        mkdir -p "$OUT"
        cp "$LOCAL"/{selection.json,history.json,run.json,config.json,progress.jsonl} "$OUT/"
        cp -r "$LOCAL/epoch-$e" "$OUT/"
        ;;
    esac
    ;;
  *) echo "usage: $0 MODEL [--set key=value ...] | test [MODEL ...]  (MODEL: configs/token-span-creddata-MODEL.yaml)" >&2; exit 2 ;;
esac
python -c "import json, sys; s = json.load(open(sys.argv[1])); print(json.dumps(dict(selected=s['selected'], history=[dict(epoch=h['epoch'], loss=round(h['loss'], 5), raw={k: h['raw'][k] for k in ('tp', 'fp', 'fn', 'f2')}, selected={k: h['selected'][k] for k in ('threshold', 'tp', 'fp', 'fn', 'precision', 'f2')}) for h in s['history']]), separators=(',', ':')))" "$OUT/selection.json"

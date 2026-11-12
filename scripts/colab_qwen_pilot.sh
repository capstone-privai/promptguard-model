#!/bin/bash
# Train the Qwen span detector (configs/qwen-span-pilot-step0-<a|b>.yaml) on the team's step-0 pilot data on a Colab
# GPU. The adapter epoch and cutoff are chosen on that model's team val only. The script prints the selection as
# one line of JSON: counts only, no text or values.
#
# Usage, from the repository folder on Colab:
#   bash scripts/colab_qwen_pilot.sh a|b
# The notebook's setup cells must already have built runs/data/span-pilot-step0-<a|b>
# (PREPARE=['bash scripts/colab_prepare_pilot_step0.sh']).
#
# Adapters, the resumable checkpoint and validation predictions go to Drive. Running the same command after a
# runtime reset continues where it stopped.
set -euo pipefail
m=$1
case $m in a|b) ;; *) echo "model must be a or b" >&2; exit 2 ;; esac
DRIVE=${DRIVE_DIR:-/content/drive/MyDrive/promptguard-runs}
OUT=$DRIVE/qwen-span-pilot-step0-$m
VAL=runs/data/span-pilot-step0-$m/validation.jsonl
CFG=configs/qwen-span-pilot-step0-$m.yaml
# Saving the optimizer state to Drive every 25 steps (the recipe's default) would write about 100 MB a minute.
# save_every only sets how often the resumable checkpoint is written; the training itself is unchanged.
# Drive keeps every overwritten version of that ~105 MB checkpoint for 30 days, and those versions count
# against the Drive quota. QWEN_SAVE_EVERY raises the interval when space runs short; a larger interval only
# means more steps to redo after a runtime reset.
SETS=(--set output_dir=$OUT/adapter --set save_every=${QWEN_SAVE_EVERY:-100})
# QWEN_DTYPE=float32|bfloat16 overrides the config's dtype for the probe, training and prediction alike.
if [ -n "${QWEN_DTYPE:-}" ]; then SETS+=(--set dtype=$QWEN_DTYPE); fi
mkdir -p "$OUT"
export QWEN_PREDICTIONS=$OUT/predictions PYTHONPATH=src USE_TF=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no GPU: stop here, do not train on CPU')"
python -c "import peft" 2>/dev/null || pip install -q "peft>=0.13,<1"
pip uninstall -q -y torchao 2>/dev/null || true  # peft rejects Colab's torchao, which is only used for quantization

python scripts/qwen_span.py prepare --train-dir runs/data/span-pilot-step0-$m --eval validation=$VAL \
  --sft-dir runs/qwen-span/pilot-step0-$m/sft --run=-pilot-step0-$m > /dev/null

if [ ! -f "$OUT/adapter/last/state.json" ]; then
  # Before hours of training, check that the longest prompt and the longest answer fit in GPU memory.
  python - "$CFG" "${QWEN_DTYPE:-}" <<'EOF'
import sys, yaml, torch
from pathlib import Path
from privai_model.qwen_span.modeling import load_model, load_tokenizer, completion_loss
from privai_model.qwen_span.train import read_sft, encode_all, lora_model
config = yaml.safe_load(Path(sys.argv[1]).read_text())
tokenizer = load_tokenizer(config['model_name'], config['revision'])
encoded = encode_all(tokenizer, read_sft(Path(config['sft_dir']) / 'train.jsonl'))
longest = max(encoded, key=lambda e: len(e['input_ids']))
longest_answer = max(encoded, key=lambda e: len(e['input_ids']) - e['prompt_length'])
dtype = sys.argv[2] or config['dtype']
# Set the model up exactly as train() does. In particular it must be in training mode: Hugging Face skips
# gradient checkpointing in eval mode, which would keep every layer's activations and overstate memory.
model = load_model(config['model_name'], dtype, config['revision'])
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
model = lora_model(model, config)
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.to('cuda').train()
print('probe dtype', dtype, flush=True)
for name, e in (('longest row', longest), ('longest answer', longest_answer)):
    torch.cuda.reset_peak_memory_stats()
    loss = completion_loss(model, torch.tensor([e['input_ids']], device='cuda'), e['prompt_length'])
    loss.backward()
    model.zero_grad(set_to_none=True)
    print(f"{name}: {len(e['input_ids'])} tokens ({len(e['input_ids']) - e['prompt_length']} answer), "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB of {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB", flush=True)
EOF
  [ -z "${PROBE_ONLY:-}" ] || exit 0
  resume=()
else
  resume=(--resume)
fi
python scripts/qwen_span.py train --config "$CFG" "${SETS[@]}" "${resume[@]}" 2>&1 | tee -a "$OUT/train.log" || true
test -f "$OUT/adapter/training.json" || { echo "training did not finish; run this script again to resume" >&2; exit 1; }

epochs=$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['epochs'])")
tags=()
for e in $(seq 1 "$epochs"); do
  python scripts/qwen_span.py predict --config "$CFG" "${SETS[@]}" --adapter "$OUT/adapter/epoch-$e" \
    --tag pilot-$m-e$e --sets validation --eval validation=$VAL | grep --line-buffered complete || true
  tags+=(pilot-$m-e$e)
done
# --force only lets a rerun in the same runtime rewrite the selection; this script scores nothing but validation.
python scripts/qwen_span.py select --tags "$(IFS=,; echo "${tags[*]}")" --run=-pilot-step0-$m --eval validation=$VAL --force
cp reports/qwen_span/selection-pilot-step0-$m.json "$OUT/"
python -c "import json, sys; print(json.dumps(dict(selection=json.load(open(sys.argv[1])), training=json.load(open(sys.argv[2]))), separators=(',', ':')))" \
  "$OUT/selection-pilot-step0-$m.json" "$OUT/adapter/training.json"

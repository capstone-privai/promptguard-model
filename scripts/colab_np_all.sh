#!/bin/bash
# Train every model again on CredData-trajectory v1 plus NoseyParker-trajectory v1 train, one step after another, on a
# Colab GPU runtime (or a local PC with DRIVE_DIR, CREDDATA_TRAJ_ZIP, NP_TRAJ_ZIP and LOCAL_RUNS set).
#
#   bash scripts/colab_np_all.sh [STEP ...]        # default: every step below, in this order
#
#   verifier-prepare           detector candidates on the combined span rows (CPU; scripts/colab_verifier.sh prepare np)
#   np-mmbert                  span model, mmBERT-small                         (scripts/colab_token_span.sh)
#   verifier-np-mmbert-small   verifier, mmBERT-small, then the validation policy comparison
#   verifier-np-mmbert-base    verifier, mmBERT-base, then the validation policy comparison
#   np-mmbert-base             span model, mmBERT-base
#   pilot-bcrednp              pilot B+CredData+NP (mmBERT-small, seed 112, as B+CredData was run)
#   np-qwen-bidir              span model, bidirectional Qwen2.5-Coder-0.5B (resumes after a reset)
#   np-qwen1.5b-bidir          span model, bidirectional Qwen2.5-Coder-1.5B (resumes after a reset)
#
# Every step fits a T4. A finished step is skipped, so after a runtime reset the same command carries on: Qwen resumes
# from its last checkpoint, mmBERT restarts the step, and an interrupted pilot run is left for
# scripts/finish_interrupted_run.py (its trainer cannot resume). A failed step does not stop the next one; failures
# are listed at the end, followed by the validation counts of every finished step.
#
# Drive promptguard-runs/ needs CredData-trajectory.zip and NoseyParker-trajectory.zip. Validation only: no step scores
# a test split here. The NP test is the team's hold-out. The pilot trainer scores B's placeholder test at its end as
# every pilot run did; that file (test-predictions.jsonl, evaluation.json) is not opened.
set -uo pipefail
DRIVE=${DRIVE_DIR:-/content/drive/MyDrive/promptguard-runs}
export DRIVE_DIR=$DRIVE PYTHONPATH=src USE_TF=0
ALL=(verifier-prepare np-mmbert verifier-np-mmbert-small verifier-np-mmbert-base np-mmbert-base pilot-bcrednp np-qwen-bidir
     np-qwen1.5b-bidir)
steps=("$@"); [ ${#steps[@]} -gt 0 ] || steps=("${ALL[@]}")
for s in "${steps[@]}"; do
  printf '%s\n' "${ALL[@]}" | grep -qx -- "$s" || { echo "unknown step: $s" >&2; exit 2; }
done
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no GPU: stop here, do not train on CPU')" || exit 1
PILOT=$DRIVE/span-mmbert-pilot-step0-bcrednp-seed112
PILOT_FINGERPRINT=9400096cf161b2231d8e7ab52f500f042728e155f5d0d2033b743ecf2d394913

finished() {
  case $1 in
    verifier-prepare) [ -f "$DRIVE/data-verifier-creddata-np-traj-v1/analysis.json" ] ;;
    verifier-*) [ -s "$DRIVE/verifier-creddata-${1#verifier-}/system_validation.json" ] ;;
    pilot-bcrednp) [ -f "$PILOT/model/span_bundle.json" ] ;;
    *) [ -f "$DRIVE/token-span-creddata-$1/selection.json" ] ;;
  esac
}

pilot() {
  if [ -d "$PILOT" ]; then
    echo "$PILOT exists without a finished bundle: an interrupted run. Its trainer cannot resume; make the bundle with" \
         "scripts/finish_interrupted_run.py instead of training again." >&2
    return 2
  fi
  bash scripts/colab_prepare_pilot_step0.sh || return
  bash scripts/colab_prepare_creddata_np_traj.sh || return
  [ -d runs/data/span-pilot-step0-bcrednp ] || python scripts/prepare_pilot_plus_ours.py bcrednp > /dev/null || return
  python -c "import json, sys; f = json.load(open('runs/data/span-pilot-step0-bcrednp/manifest.json'))['fingerprint']; print('pilot data fingerprint', f); sys.exit(0 if f == '$PILOT_FINGERPRINT' else 'pilot data differs from the local build')" || return
  pip install -q -e '.[baselines]' || return  # the pilot trainer ends with the rule baseline (CredSweeper)
  python -c "from huggingface_hub import snapshot_download; snapshot_download('jhu-clsp/mmBERT-small', revision='abc32620dd4f6ab06f5fbe905dc25f310618e09f', local_dir='runs/backbones/mmbert-small', allow_patterns=['*.json', 'pytorch_model.bin'])" || return
  python - "$PILOT" <<'EOF' || return
import subprocess, sys
import torch, yaml
base = 'configs/span-pilot-step0-bcrednp.yaml'
config = yaml.safe_load(open(base))
config.update(model_name='runs/backbones/mmbert-small', device='cuda', local_files_only=True, seed=112, output_dir=sys.argv[1])
config['colab'] = dict(gpu=torch.cuda.get_device_name(0), start_from='backbone', base_config=base, overrides=dict(seed=112),
                       commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                       torch=str(torch.__version__))
with open('runs/pilot-bcrednp.yaml', 'w') as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
EOF
  python scripts/span_pipeline.py train --config runs/pilot-bcrednp.yaml 2>&1 | grep --line-buffered -v -i warn | tee "$PILOT.log"
}

run() {
  case $1 in
    verifier-prepare) bash scripts/colab_verifier.sh prepare np ;;
    verifier-*)
      m=${1#verifier-}
      bash scripts/colab_verifier.sh prepare np || return
      [ -f "$DRIVE/verifier-creddata-$m/model/bundle.json" ] || bash scripts/colab_verifier.sh train "$m" || return
      bash scripts/colab_verifier.sh system "$m"
      ;;
    pilot-bcrednp) pilot ;;
    *) bash scripts/colab_token_span.sh "$1" ;;
  esac
}

# Publishing: each finished mmBERT model (span, verifier, pilot) becomes a models/ bundle with its weights stored in
# bfloat16 (scripts/half_bundle.py, validation rescored with both copies), packed as $PUB/<name>.tar for Git LFS.
# The Qwen runs stay on Drive as they are.
PUB=$DRIVE/publish-np
publish() {
  local name src check
  case $1 in
    np-mmbert) name=span-creddata-np-mmbert-small ;;
    np-mmbert-base) name=span-creddata-np-mmbert-base ;;
    verifier-np-mmbert-small|verifier-np-mmbert-base) name=${1#verifier-np-}; name=verifier-creddata-np-$name ;;
    pilot-bcrednp) name=span-pilot-step0-bcrednp-seed112 ;;
    *) return 0 ;;
  esac
  [ -f "$PUB/$name.tar" ] && return 0
  case $1 in  # the validation data the check needs (a fresh runtime after a reset has none)
    np-mmbert*) bash scripts/colab_prepare_creddata_np_traj.sh > /dev/null || return ;;
    verifier-*) bash scripts/colab_verifier.sh prepare np > /dev/null || return ;;
    pilot-bcrednp)
      [ -d runs/data/span-pilot-step0-bcrednp ] || { bash scripts/colab_prepare_pilot_step0.sh > /dev/null \
        && bash scripts/colab_prepare_creddata_np_traj.sh > /dev/null \
        && python scripts/prepare_pilot_plus_ours.py bcrednp > /dev/null; } || return ;;
  esac
  case $1 in
    np-mmbert*)
      src=/content/full-$1; rm -rf "$src"
      python scripts/export_token_span_bundle.py --run "$DRIVE/token-span-creddata-$1" --out "$src" || return
      check=(--check-span runs/data/span-creddata-np-traj-v1/validation.jsonl) ;;
    verifier-*)
      src=$DRIVE/verifier-creddata-${1#verifier-}/model
      check=(--check-verifier runs/data/verifier-creddata-np-traj-v1/ambiguous/candidates.jsonl.gz) ;;
    pilot-bcrednp)
      src=$PILOT/model
      check=(--check-span runs/data/span-pilot-step0-bcrednp/validation.jsonl) ;;
  esac
  rm -rf "/content/pub/$name" && mkdir -p /content/pub "$PUB"
  python scripts/half_bundle.py --src "$src" --out "/content/pub/$name" "${check[@]}" --device cuda || return
  tar -cf "$PUB/$name.tar.tmp" -C /content/pub "$name" && mv "$PUB/$name.tar.tmp" "$PUB/$name.tar"
  echo "published $PUB/$name.tar ($(du -h "$PUB/$name.tar" | cut -f1))"
}

failed=()
for s in "${steps[@]}"; do
  if finished "$s"; then echo "== $s: finished, skipped"
  else
    echo "== $s  ($(date -u +%FT%TZ))"
    run "$s" || { failed+=("$s"); echo "!! $s failed" >&2; continue; }
  fi
  publish "$s" || { failed+=("publish $s"); echo "!! publishing $s failed" >&2; }
done
echo "failed steps: ${failed[*]:-none}"

# Validation counts of every finished step (no text, no values; test files are not read).
python - "$DRIVE" "$PILOT" <<'EOF'
import json, os, sys
drive, pilot = sys.argv[1], sys.argv[2]
keys = ('threshold', 'tp', 'fp', 'fn', 'precision', 'recall', 'f2')
short = lambda m: {k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in keys if k in m}
for m in ('np-mmbert', 'np-mmbert-base', 'np-qwen-bidir', 'np-qwen1.5b-bidir'):
    path = f'{drive}/token-span-creddata-{m}/selection.json'
    if not os.path.exists(path):
        print(f'== span {m}: not finished'); continue
    s = json.load(open(path))
    for h in s['history']:
        mark = '*' if h['epoch'] == s['selected']['epoch'] else ' '
        print(f"== span {m} epoch {h['epoch']}{mark}", json.dumps(short(h['selected'])),
              json.dumps({n: short(v) for n, v in h.get('by_source', {}).items()}))
for m in ('np-mmbert-small', 'np-mmbert-base'):
    path = f'{drive}/verifier-creddata-{m}/system_validation.json'
    if not os.path.exists(path) or not os.path.getsize(path):
        print(f'== verifier {m}: not finished'); continue
    s = json.load(open(path))
    pick = lambda d: {p: {k: d[p][k] for k in ('masked', 'false_masks', 'gold_reached', 'precision', 'recall', 'f2')}
                      for p in ('union', 'strong_only', 'strong_verifier')}
    print(f'== verifier {m} threshold {round(s["threshold"], 4)}', json.dumps(pick(s)))
    for n, d in s.get('by_source', {}).items():
        print(f'   {n}', json.dumps(pick(d)))
if os.path.exists(f'{pilot}/history.json'):
    for h in json.load(open(f'{pilot}/history.json')):
        v = h['validation']
        print(f"== pilot bcrednp epoch {h['epoch']} cutoff {h['threshold']}", json.dumps(short(v)))
else:
    print('== pilot bcrednp: not started')
EOF
[ ${#failed[@]} -eq 0 ]

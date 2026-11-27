#!/bin/bash
# Train the two models chosen for Trajectory-train-combined v1 (the team's third training set: CredData, NoseyParker
# and infra-config trajectories) one after another on a Colab GPU runtime (or a local PC with DRIVE_DIR and the zips
# set, see scripts/colab_prepare_traj_combined.sh):
#
#   bash scripts/colab_tc_all.sh [STEP ...]        # default: every step below, in this order
#
#   verifier-prepare           detector candidates on the combined span rows (CPU; scripts/colab_verifier.sh prepare tc)
#   verifier-tc-mmbert-small   verifier, mmBERT-small, then the validation policy comparison
#   tc-mmbert-base             span model, mmBERT-base                          (scripts/colab_token_span.sh)
#
# A finished step is skipped, so after a runtime reset the same command carries on (mmBERT restarts the step). A failed
# step does not stop the next one; failures are listed at the end, followed by the validation counts of every finished
# step. Each finished model is packed as a models/ bundle with float16 weights (scripts/half_bundle.py; validation is
# rescored with both copies) in $DRIVE/publish-tc/<name>.tar.
#
# Drive promptguard-runs/ needs CredData-trajectory.zip, NoseyParker-trajectory.zip and Trajectory-train-combined.zip.
# Validation only: no step scores a test split. The NoseyParker and infra-config tests are the team's hold-outs, and the
# CredData test groups stay closed.
set -uo pipefail
DRIVE=${DRIVE_DIR:-/content/drive/MyDrive/promptguard-runs}
export DRIVE_DIR=$DRIVE PYTHONPATH=src USE_TF=0
ALL=(verifier-prepare verifier-tc-mmbert-small tc-mmbert-base)
steps=("$@"); [ ${#steps[@]} -gt 0 ] || steps=("${ALL[@]}")
for s in "${steps[@]}"; do
  printf '%s\n' "${ALL[@]}" | grep -qx -- "$s" || { echo "unknown step: $s" >&2; exit 2; }
done
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no GPU: stop here, do not train on CPU')" || exit 1

finished() {
  case $1 in
    verifier-prepare) [ -f "$DRIVE/data-verifier-traj-combined-v1/analysis.json" ] ;;
    verifier-tc-mmbert-small) [ -s "$DRIVE/verifier-traj-combined-mmbert-small/system_validation.json" ] ;;
    tc-mmbert-base) [ -f "$DRIVE/token-span-traj-combined-mmbert-base/selection.json" ] ;;
  esac
}

run() {
  case $1 in
    verifier-prepare) bash scripts/colab_verifier.sh prepare tc ;;
    verifier-tc-mmbert-small)
      bash scripts/colab_verifier.sh prepare tc || return
      [ -f "$DRIVE/verifier-traj-combined-mmbert-small/model/bundle.json" ] \
        || bash scripts/colab_verifier.sh train tc-mmbert-small || return
      bash scripts/colab_verifier.sh system tc-mmbert-small
      ;;
    tc-mmbert-base) bash scripts/colab_token_span.sh tc-mmbert-base ;;
  esac
}

PUB=$DRIVE/publish-tc
publish() {
  local name src check
  case $1 in
    verifier-tc-mmbert-small) name=verifier-traj-combined-mmbert-small ;;
    tc-mmbert-base) name=span-traj-combined-mmbert-base ;;
    *) return 0 ;;
  esac
  [ -f "$PUB/$name.tar" ] && return 0
  case $1 in  # the validation data the check needs (a fresh runtime after a reset has none)
    tc-*) bash scripts/colab_prepare_traj_combined.sh > /dev/null || return
      src=/content/full-$1; rm -rf "$src"
      python scripts/export_token_span_bundle.py --run "$DRIVE/token-span-traj-combined-${1#tc-}" --out "$src" || return
      check=(--check-span runs/data/span-traj-combined-v1/validation.jsonl) ;;
    verifier-*) bash scripts/colab_verifier.sh prepare tc > /dev/null || return
      src=$DRIVE/verifier-traj-combined-mmbert-small/model
      check=(--check-verifier runs/data/verifier-traj-combined-v1/ambiguous/candidates.jsonl.gz) ;;
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
python - "$DRIVE" <<'EOF'
import json, os, sys
drive = sys.argv[1]
keys = ('threshold', 'tp', 'fp', 'fn', 'precision', 'recall', 'f2')
short = lambda m: {k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in keys if k in m}
path = f'{drive}/token-span-traj-combined-mmbert-base/selection.json'
if os.path.exists(path):
    s = json.load(open(path))
    for h in s['history']:
        mark = '*' if h['epoch'] == s['selected']['epoch'] else ' '
        print(f"== span tc-mmbert-base epoch {h['epoch']}{mark} loss {round(h['loss'], 5)}", json.dumps(short(h['selected'])),
              json.dumps({n: short(v) for n, v in h.get('by_source', {}).items()}))
else:
    print('== span tc-mmbert-base: not finished')
path = f'{drive}/verifier-traj-combined-mmbert-small/system_validation.json'
if os.path.exists(path) and os.path.getsize(path):
    s = json.load(open(path))
    pick = lambda d: {p: {k: d[p][k] for k in ('masked', 'false_masks', 'gold_reached', 'precision', 'recall', 'f2')}
                      for p in ('union', 'strong_only', 'strong_verifier')}
    print(f'== verifier tc-mmbert-small threshold {round(s["threshold"], 4)}', json.dumps(pick(s)))
    for n, d in s.get('by_source', {}).items():
        print(f'   {n}', json.dumps(pick(d)))
else:
    print('== verifier tc-mmbert-small: not finished')
EOF
[ ${#failed[@]} -eq 0 ]

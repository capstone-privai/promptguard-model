#!/bin/bash
# Re-make the publish tar of one finished NP step on Colab (the same steps as publish() in colab_np_all.sh), e.g.
# after half_bundle.py changed. Runs beside a training chain; the chain script itself is not touched.
#
#   bash scripts/colab_np_publish.sh [--force] STEP     # STEP: np-mmbert, np-mmbert-base, verifier-np-mmbert-small,
#                                                       #       verifier-np-mmbert-base, pilot-bcrednp
# --force replaces an existing $DRIVE_DIR/publish-np/<name>.tar. Prints the half_storage check and ends with PUBLISHED.
set -euo pipefail
force=0; [ "${1:-}" = --force ] && { force=1; shift; }
step=${1:?usage: $0 [--force] STEP}
DRIVE=${DRIVE_DIR:-/content/drive/MyDrive/promptguard-runs}
PUB=$DRIVE/publish-np
PILOT=$DRIVE/span-mmbert-pilot-step0-bcrednp-seed112
export PYTHONPATH=src USE_TF=0
case $step in
  np-mmbert) name=span-creddata-np-mmbert-small ;;
  np-mmbert-base) name=span-creddata-np-mmbert-base ;;
  verifier-np-mmbert-small|verifier-np-mmbert-base) name=verifier-creddata-np-${step#verifier-np-} ;;
  pilot-bcrednp) name=span-pilot-step0-bcrednp-seed112 ;;
  *) echo "nothing to publish for $step" >&2; exit 2 ;;
esac
if [ -f "$PUB/$name.tar" ] && [ $force = 0 ]; then echo "$PUB/$name.tar exists (use --force)"; exit 0; fi
case $step in
  np-mmbert*) bash scripts/colab_prepare_creddata_np_traj.sh > /dev/null ;;
  verifier-*) bash scripts/colab_verifier.sh prepare np > /dev/null ;;
  pilot-bcrednp)
    [ -d runs/data/span-pilot-step0-bcrednp ] || { bash scripts/colab_prepare_pilot_step0.sh > /dev/null \
      && bash scripts/colab_prepare_creddata_np_traj.sh > /dev/null \
      && python scripts/prepare_pilot_plus_ours.py bcrednp > /dev/null; } ;;
esac
case $step in
  np-mmbert*)
    src=/content/full-$step; rm -rf "$src"
    python scripts/export_token_span_bundle.py --run "$DRIVE/token-span-creddata-$step" --out "$src"
    check=(--check-span runs/data/span-creddata-np-traj-v1/validation.jsonl) ;;
  verifier-*)
    src=$DRIVE/verifier-creddata-${step#verifier-}/model
    check=(--check-verifier runs/data/verifier-creddata-np-traj-v1/ambiguous/candidates.jsonl.gz) ;;
  pilot-bcrednp)
    src=$PILOT/model
    check=(--check-span runs/data/span-pilot-step0-bcrednp/validation.jsonl) ;;
esac
work=/content/pub-redo; rm -rf "$work/$name" && mkdir -p "$work" "$PUB"
python scripts/half_bundle.py --src "$src" --out "$work/$name" "${check[@]}" --device cuda
tar -cf "$PUB/$name.tar.tmp" -C "$work" "$name" && mv "$PUB/$name.tar.tmp" "$PUB/$name.tar"
echo "PUBLISHED $PUB/$name.tar ($(du -h "$PUB/$name.tar" | cut -f1))"

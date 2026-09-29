#!/usr/bin/env bash
# v2 training, unattended: data -> train -> test scoring -> upload. Run inside tmux from the
# folder holding gamus/, SynRS3D/ and runs/ft/best.pth (the v1 model it continues from):
#
#     bash scripts/overnight_v2.sh 2>&1 | tee overnight.log
#
# Every stage runs even if an earlier one partly failed, so the night is never wasted:
# downloads skip failures, training falls back to a smaller batch, scoring and upload
# use whichever v2 checkpoint exists.
set -u
HOURS="${HOURS:-6.5}"
REPO="${REPO:-Dilavesh/altimap-height}"
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1
stamp() { echo "=== $(date '+%H:%M:%S') $*"; }

stamp "1/4 data"
python scripts/fetch_training_data.py

train() {  # $1 = batch, $2 = hours; resumes from v2 progress if an earlier attempt died
  local ckpt=runs/ft/best.pth
  [ -f runs/v2/best.pth ] && ckpt=runs/v2/best.pth
  python -m viewer.height_train --ckpt "$ckpt" --out runs/v2 --hours "$2" \
    --batch "$1" --unfreeze-blocks 8 --lr 5e-5 --encoder-lr 1e-5 --height-weight 10 \
    --syn-data synrs3d --syn-fraction 0.3 --degrade 0.5 --workers 10
}
stamp "2/4 train (${HOURS} h)"
train 8 "$HOURS" || { stamp "batch 8 failed; retrying with batch 4"; train 4 5; }

stamp "3/4 test scoring"
if [ -f runs/v2/best.pth ]; then
  python -m viewer.height_eval --split test --tta --ckpt runs/v2/best.pth --out test_v2.json 2>&1 \
    | tee test_v2.log
fi

stamp "4/4 upload"
for f in runs/v2/best.pth runs/v2/best.json runs/v2/log.jsonl test_v2.json test_v2.log overnight.log; do
  [ -f "$f" ] && hf upload "$REPO" "$f" "v2/$(basename "$f")" --private || true
done
stamp "finished"

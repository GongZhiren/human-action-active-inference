#!/bin/bash
# Clean final experiment matrix: 4 datasets x 3 seeds, seeds concurrent per dataset.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TAG=${1:-}
EPOCHS=${2:-15}
DATASETS=${3:-"car nba ddxplus atari"}

run_dataset () {
  local ds=$1
  echo "=== $ds : 3 seeds ==="
  for s in 0 1 2; do
    python3 scripts/run.py --dataset "$ds" --seed "$s" --epochs "$EPOCHS" --tag "$TAG" \
      > "logs/${ds}_s${s}${TAG:+_$TAG}.out" 2>&1 &
  done
  wait
  echo "=== $ds done ==="
}

for ds in $DATASETS; do
  run_dataset "$ds"
done
echo "ALL DATASETS COMPLETE"

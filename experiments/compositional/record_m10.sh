#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ROBOSTEP_PYTHON:-python}"
CHECKPOINT="${1:-$ROOT/artifacts/checkpoints/compositional_m10/model_6999.pt}"
OUTPUT_DIR="${2:-$ROOT/runs/compositional/m10_video}"
VIDEO="${ROBOSTEP_M10_VIDEO:-$OUTPUT_DIR/m10_rule_policy.mp4}"
export PYTHONPATH="$ROOT/main_method/metaworld:$ROOT/main_method:$ROOT/main_method/stage_policy:${PYTHONPATH:-}"

exec "$PYTHON_BIN" -m stage_reward.eval_bidirectional_pick_place "$CHECKPOINT" \
  --gate rule \
  --episodes "${ROBOSTEP_M10_EPISODES:-1}" \
  --num-envs 1 \
  --seed "${ROBOSTEP_M10_SEED:-1042}" \
  --output "$OUTPUT_DIR" \
  --video "$VIDEO" \
  --video-fps "${ROBOSTEP_M10_VIDEO_FPS:-20}"

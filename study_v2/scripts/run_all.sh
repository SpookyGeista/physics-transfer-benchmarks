#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
STUDY_PY=/home/loya2/miniconda3/envs/unitree_sim_env/bin/python
STUDY_MJ=/home/loya2/miniconda3/envs/mujoco/bin/python
mkdir -p results
"$STUDY_PY" -m pytest -q tests
if ! "$STUDY_PY" -c 'from pathlib import Path; assert len(list(Path("results/data").glob("*/*/privileged.json")))==10'; then
 "$STUDY_PY" scripts/generate.py --headless --device cpu --stage calibration
fi
"$STUDY_PY" scripts/train.py
if [ ! -f results/protocol_lock.json ]; then
 "$STUDY_PY" scripts/support.py --stage fit
fi
"$STUDY_PY" scripts/freeze.py
if ! "$STUDY_PY" -c 'from pathlib import Path; assert len(list(Path("results/data").glob("*/*/replay.json")))==10'; then
 "$STUDY_PY" scripts/generate.py --headless --device cpu --stage replay
fi
"$STUDY_PY" scripts/support.py --stage evaluate
"$STUDY_PY" scripts/evaluate.py
for substeps in 1 2 4; do
 if [ ! -f "results/mujoco/boucwen_r0_h${substeps}.json" ] || [ ! -f "results/mujoco/maxwell_r0_h${substeps}.json" ]; then
  "$STUDY_MJ" scripts/mujoco_replay.py --seeds 0 --substeps "$substeps"
 fi
done
"$STUDY_PY" scripts/control.py --headless --device cpu
"$STUDY_PY" scripts/support.py --stage control
"$STUDY_PY" scripts/report.py
bash supplement_exact/scripts/run_all.sh
"$STUDY_PY" scripts/conclusion.py
for world in maxwell boucwen; do
 for task in replay control; do
  if [ ! -f "videos/${world}_${task}.mp4" ]; then
   "$STUDY_PY" scripts/render_video.py --headless --enable_cameras --device cpu --world "$world" --task "$task" --study "$(if [ "$task" = replay ]; then echo supplement; else echo original; fi)"
  fi
 done
done
"$STUDY_PY" scripts/verify.py

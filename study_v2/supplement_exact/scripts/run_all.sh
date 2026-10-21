#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
STUDY_PY=/home/loya2/miniconda3/envs/unitree_sim_env/bin/python
STUDY_MJ=/home/loya2/miniconda3/envs/mujoco/bin/python
if [ ! -f supplement_exact/lock.json ]; then
 "$STUDY_PY" supplement_exact/scripts/freeze.py
fi
if ! "$STUDY_PY" -c 'from pathlib import Path; assert len(list(Path("supplement_exact/results/data").glob("*/*/replay.json")))==10'; then
 "$STUDY_PY" supplement_exact/scripts/generate.py --headless --device cpu
fi
"$STUDY_PY" supplement_exact/scripts/audit.py
"$STUDY_PY" supplement_exact/scripts/evaluate.py
for substeps in 1 2 4; do
 if [ ! -f "supplement_exact/results/mujoco/boucwen_r0_h${substeps}.json" ] || [ ! -f "supplement_exact/results/mujoco/maxwell_r0_h${substeps}.json" ]; then
  "$STUDY_MJ" scripts/mujoco_replay.py --seeds 0 --substeps "$substeps" --input supplement_exact/results/data --out supplement_exact/results/mujoco
 fi
done
"$STUDY_PY" supplement_exact/scripts/analyze.py

#!/usr/bin/env bash
# Resume completed stages; run from a clean copied directory for a fresh reproduction.
set -euo pipefail
cd "$(dirname "$0")/.."
NPT_PYTHON=${NPT_PYTHON:-/home/loya2/miniconda3/envs/unitree_sim_env/bin/python}
NPT_MUJOCO=${NPT_MUJOCO:-/home/loya2/miniconda3/envs/mujoco/bin/python}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/null-physics-transfer-matplotlib}
mkdir -p results videos
if [ ! -f results/isaac_smoke.json ]; then "$NPT_PYTHON" scripts/isaac_smoke.py --headless > results/isaac_smoke.log 2>&1; fi
if [ ! -f results/data/nonlinear_privileged.npz ]; then "$NPT_PYTHON" scripts/generate.py --headless --device cpu --splits train val privileged > results/generate.log 2>&1; fi
"$NPT_PYTHON" scripts/data_audit.py > results/data_audit.log
"$NPT_PYTHON" -m pytest -q tests
"$NPT_PYTHON" scripts/train.py > results/train_resume.log
if [ ! -f results/protocol_lock.json ]; then "$NPT_PYTHON" scripts/freeze.py; fi
if [ ! -f results/data/sls_test.npz ]; then "$NPT_PYTHON" scripts/generate.py --headless --device cpu --splits test > results/generate_test.log 2>&1; fi
if [ ! -f results/evaluation/prediction.csv ]; then "$NPT_PYTHON" scripts/evaluate.py > results/evaluate.log; fi
if [ ! -f results/control/control.csv ]; then "$NPT_PYTHON" scripts/control.py --headless --device cpu > results/control.log 2>&1; fi
if [ ! -f results/mujoco/sls_test.npz ]; then "$NPT_MUJOCO" scripts/mujoco_reference.py > results/mujoco.log; fi
if [ ! -f results/mujoco_half/sls_test.npz ]; then "$NPT_MUJOCO" scripts/mujoco_reference.py --substeps 2 --out results/mujoco_half > results/mujoco_half.log; fi
if [ ! -f results/mujoco_quarter/sls_test.npz ]; then "$NPT_MUJOCO" scripts/mujoco_reference.py --substeps 4 --out results/mujoco_quarter > results/mujoco_quarter.log; fi
if [ ! -f results/mujoco_native/sls_test.npz ]; then "$NPT_MUJOCO" scripts/mujoco_reference.py --native-friction --out results/mujoco_native > results/mujoco_native.log; fi
if [ ! -f results/evaluation_native/prediction.csv ]; then "$NPT_PYTHON" scripts/evaluate.py --data results/mujoco_native --out results/evaluation_native > results/evaluate_native.log; fi
for world in sls nonlinear; do
 if [ ! -f "videos/${world}_isaac_replay.mp4" ]; then "$NPT_PYTHON" scripts/render_video.py --headless --enable_cameras --world "$world" > "results/render_video_${world}.log" 2>&1; fi
done
if [ ! -f results/environment.json ]; then "$NPT_PYTHON" scripts/environment.py > results/environment_inventory.log; fi
"$NPT_PYTHON" scripts/report.py
"$NPT_PYTHON" scripts/verify.py

#!/usr/bin/env bash
# Technical repetition of the fixed study; preserves all original outputs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -ne 1 || ${1:-} == --help ]]; then
  echo 'Usage: bash matched_study/reproduce.sh UNIQUE_TAG'
  echo 'Runs the fixed main study, four controls, analyses, audits and video in fresh paths.'
  exit 0
fi
matched_tag=$1
if [[ ! $matched_tag =~ ^[A-Za-z][A-Za-z0-9_-]*$ ]]; then
  echo 'Tag must start with a letter; only letters, digits, underscores or hyphens allowed.' >&2
  exit 2
fi
matched_python=${MATCHED_PYTHON:-/home/loya2/miniconda3/envs/unitree_sim_env/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR=/tmp/physics-transfer-matched-numba
export MPLCONFIGDIR=/tmp/physics-transfer-matched-matplotlib
matched_base=matched_study/results
for matched_name in main h2 h4 p256 p1024 controls; do
  if [[ -e "$matched_base/${matched_name}_${matched_tag}" || -e "matched_study/reports/audit.${matched_name}_${matched_tag}.json" ]]; then
    echo "Output already exists for ${matched_name}_${matched_tag}; choose a fresh tag." >&2
    exit 2
  fi
done
"$matched_python" matched_study/run.py freeze --kind main --out "$matched_base/main_${matched_tag}"
"$matched_python" matched_study/run.py run --out "$matched_base/main_${matched_tag}"
"$matched_python" matched_study/analyze.py --data "$matched_base/main_${matched_tag}" --out "$matched_base/main_${matched_tag}/analysis"
"$matched_python" matched_study/audit.py run --data "$matched_base/main_${matched_tag}" --out "matched_study/reports/audit.main_${matched_tag}"
for matched_pair in h2:0.00125 h4:0.000625; do
  matched_name=${matched_pair%%:*}
  matched_step=${matched_pair#*:}
  "$matched_python" matched_study/run.py freeze --kind refinement --dt "$matched_step" --out "$matched_base/${matched_name}_${matched_tag}"
  "$matched_python" matched_study/run.py run --out "$matched_base/${matched_name}_${matched_tag}"
  "$matched_python" matched_study/audit.py run --data "$matched_base/${matched_name}_${matched_tag}" --out "matched_study/reports/audit.${matched_name}_${matched_tag}"
done
for matched_particles in 256 1024; do
  "$matched_python" matched_study/run.py freeze --kind resolution --particles "$matched_particles" --out "$matched_base/p${matched_particles}_${matched_tag}"
  "$matched_python" matched_study/run.py run --out "$matched_base/p${matched_particles}_${matched_tag}"
  "$matched_python" matched_study/audit.py run --data "$matched_base/p${matched_particles}_${matched_tag}" --out "matched_study/reports/audit.p${matched_particles}_${matched_tag}"
done
"$matched_python" matched_study/compare_controls_v2.py \
  --main "$matched_base/main_${matched_tag}" \
  --refinement "$matched_base/h2_${matched_tag}" "$matched_base/h4_${matched_tag}" \
  --resolution "$matched_base/p256_${matched_tag}" "$matched_base/p1024_${matched_tag}" \
  --out "$matched_base/controls_${matched_tag}"
"$matched_python" matched_study/make_video.py --data "$matched_base/main_${matched_tag}" --out "$matched_base/main_${matched_tag}/video"
for matched_name in main h2 h4 p256 p1024; do
  "$matched_python" matched_study/run.py verify --out "$matched_base/${matched_name}_${matched_tag}"
done
echo "Complete: $matched_base/main_${matched_tag}/analysis/REPORT.md"
echo "Controls: $matched_base/controls_${matched_tag}/REPORT.md"

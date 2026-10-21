#!/usr/bin/env bash
# Rerun frozen experiments in the existing environment; no installation/network.
set -euo pipefail
PAPER_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$PAPER_REPO"
PAPER_PYTHON=${PAPER_PYTHON:-/home/loya2/miniconda3/envs/unitree_sim_env/bin/python}
PAPER_STUDY=${1:?Use main1, main2 or main3, followed by a unique output suffix}
PAPER_RUN=${2:?Provide a unique alphanumeric output suffix}
if [[ ! "$PAPER_RUN" =~ ^[a-zA-Z0-9_]+$ ]]; then
    exit 2
fi
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR=/tmp/paper-study-numba
export MPLCONFIGDIR=/tmp/paper-matplotlib
case "$PAPER_STUDY" in
main1)
    PAPER_NATIVE="paper_study/results/native_main1_${PAPER_RUN}"
    PAPER_EVAL="paper_study/results/observer_main1_${PAPER_RUN}"
    PAPER_STATS="paper_study/results/statistics_main1_${PAPER_RUN}"
    test ! -e "$PAPER_NATIVE"
    test ! -e "$PAPER_EVAL"
    test ! -e "$PAPER_STATS"
    "$PAPER_PYTHON" paper_study/scripts/generate_native_v2.py \
        --config paper_study/configs/main1_native.json --headless --device cpu \
        --test-lock paper_study/locks/main1/test_authorization.json --out "$PAPER_NATIVE"
    "$PAPER_PYTHON" paper_study/scripts/evaluate_native_v2.py \
        --config paper_study/configs/main1_observer.json --data "$PAPER_NATIVE" --out "$PAPER_EVAL"
    "$PAPER_PYTHON" paper_study/scripts/main1_statistics.py --data "$PAPER_EVAL" --out "$PAPER_STATS"
    ;;
main2)
    PAPER_NATIVE="paper_study/results/native_main2_${PAPER_RUN}"
    PAPER_EVAL="paper_study/results/observer_main2_${PAPER_RUN}"
    test ! -e "$PAPER_NATIVE"
    test ! -e "$PAPER_EVAL"
    "$PAPER_PYTHON" paper_study/costed_benchmark/generate_main2.py --headless --device cpu \
        --test-lock paper_study/locks/main2/test_authorization.json --out "$PAPER_NATIVE"
    "$PAPER_PYTHON" paper_study/costed_benchmark/evaluate_main2.py freeze \
        --data "$PAPER_NATIVE" --test-lock paper_study/locks/main2/test_authorization.json --out "$PAPER_EVAL"
    for PAPER_BANK in 1 2 3; do
        "$PAPER_PYTHON" paper_study/costed_benchmark/evaluate_main2.py run --out "$PAPER_EVAL" --bank "$PAPER_BANK"
    done
    "$PAPER_PYTHON" paper_study/costed_benchmark/evaluate_main2.py verify --out "$PAPER_EVAL"
    ;;
main3)
    export NUMBA_CACHE_DIR=/tmp/paper-decision-test-numba
    PAPER_NATIVE="paper_study/results/decision_test_native_${PAPER_RUN}"
    PAPER_EVAL="paper_study/results/decision_test_evaluation_${PAPER_RUN}"
    test ! -e "$PAPER_NATIVE"
    test ! -e "$PAPER_EVAL"
    "$PAPER_PYTHON" -m paper_study.decision_test.generate --headless --device cpu \
        --test-lock paper_study/results/decision_test_authorization.json --out "$PAPER_NATIVE"
    "$PAPER_PYTHON" -m paper_study.decision_test.evaluate freeze \
        --test-lock paper_study/results/decision_test_authorization.json --data "$PAPER_NATIVE" --out "$PAPER_EVAL"
    for PAPER_BANK in 1 2 3 4 5 6 7 8; do
        "$PAPER_PYTHON" -m paper_study.decision_test.evaluate run-bank --out "$PAPER_EVAL" --bank "$PAPER_BANK"
    done
    "$PAPER_PYTHON" -m paper_study.decision_test.evaluate aggregate --out "$PAPER_EVAL"
    ;;
*)
    exit 2
    ;;
esac

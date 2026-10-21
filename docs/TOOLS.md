# Physics experiment tools

Use the [README setup](../README.md) first. Commands run from the repository root. Choose the Isaac environment's Python for the simulator stages; CPU training/analysis use their corresponding scientific dependencies.

## Original experiment stages

| Step | Command | Required input / output |
|---|---|---|
| Simulator check | `python scripts/isaac_smoke.py --headless --device cpu` | Writes `results/isaac_smoke.json`. |
| Calibration data | `python scripts/generate.py --headless --device cpu --splits train val privileged` | Writes trajectories under `results/data/`. |
| Data audit | `python scripts/data_audit.py` | Checks generated calibration data. |
| Training | `python scripts/train.py` | Uses calibration data; writes `results/models/`. |
| Freeze | `python scripts/freeze.py` | Locks the trained state before test generation. |
| Test data | `python scripts/generate.py --headless --device cpu --splits test` | Requires the frozen protocol state. |
| Evaluation | `python scripts/evaluate.py` | Writes `results/evaluation/`. |
| MuJoCo reference | `python scripts/mujoco_reference.py` | Run using the MuJoCo environment. |
| Final verification | `python scripts/verify.py` | Requires completed numerical artifacts. |

The complete sequence, including control experiments, alternate integration steps, reports, and renders, is implemented by:

```bash
export NPT_PYTHON=/path/to/isaac-environment/bin/python
export NPT_MUJOCO=/path/to/mujoco-environment/bin/python
mkdir -p results reports videos
bash scripts/run_all.sh
```

The wrapper resumes existing stages; start a new reproduction in a clean checkout. Do not generate test data before freezing the intended model state.

## Small data-generation smoke

```bash
python scripts/generate.py --headless --device cpu \
  --splits train --count 2 --steps 80 --out results/smoke_data
```

This checks generation with a small calibration batch. It does not replace the full training/evaluation sequence.

## Controlled follow-up

```bash
bash matched_study/reproduce.sh --help
```

The follow-up runner provides `freeze`, `run`, and `verify` stages. Full reproduction also needs its fitted calibration banks, protocol, and preflight records, which are not included in this export. The saved CSVs and figure under `evidence/` can be inspected immediately.

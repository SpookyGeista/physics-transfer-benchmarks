# Physics Transfer Benchmarks

Simulation experiments on friction, compliance, and material memory.

- Separate and joint prediction models with analytical references.
- Controlled experiments on measurement information and state alignment.
- Validation, evaluation, and plots.

## Usage

Models are in `src/`, initial experiments in `scripts/`, and the controlled follow-up in `matched_study/`.

```bash
PYTHONPATH=src python -m pytest -q tests
```

The follow-up uses synthetic systems with matched truth and estimation laws.
The initial experiment found no consistent advantage for modular learning.
Selected saved results are in `evidence/`.

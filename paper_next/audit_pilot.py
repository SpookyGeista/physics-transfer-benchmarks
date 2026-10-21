"""Independent numerical read-back of the new material pilot and its hashes."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "pilot_memory/results"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    run = json.loads((DATA / "run.json").read_text())
    assert run["status"] == "complete"
    for name, digest in run["source_sha256"].items():
        assert sha(name) == digest, name
    for name, digest in run["outputs_sha256"].items():
        assert sha(DATA / name) == digest, name
    table = pd.read_csv(DATA / "trials.csv")
    assert len(table) == run["rows"] == 25920
    assert np.isfinite(table.select_dtypes(include="number").values).all()
    raw_checks = 0
    physical = []
    for world in ("maxwell", "boucwen"):
        for kind in ("slip", "rest"):
            with np.load(DATA / f"{world}_{kind}_raw.npz") as raw:
                for di, duration in enumerate(raw["durations"]):
                    # Physical response variation is distinct from predictor RMSE.
                    forces = raw["true_future_force"][di]
                    physical.append({"world": world, "probe_kind": kind,
                                     "duration_s": float(duration),
                                     "true_future_force_sd_rms_N": float(np.sqrt(np.var(forces, axis=0).mean())),
                                     "true_current_state_span": float(np.ptp(raw["true_current"][di]))})
                    for si, sigma in enumerate(raw["force_sigmas"]):
                        weights = raw["sensed_posterior"][di, si]
                        np.testing.assert_allclose(weights.sum(axis=-1), 1., atol=1e-13)
                        if duration == 0 or kind == "rest":
                            np.testing.assert_array_equal(weights, np.full_like(weights, 1 / weights.shape[-1]))
                        predictions = np.einsum("irg,gt->irt", weights, raw["candidate_future_force"][di])
                        mse = np.mean((predictions - forces[:, None, :]) ** 2, axis=-1)
                        chosen = table[(table.world == world) & (table.probe_kind == kind)
                                       & (table.duration_s == duration) & (table.sigma_N == sigma)
                                       & (table.condition == "sensed")].sort_values(["initial_case", "noise_repeat"])
                        np.testing.assert_allclose(chosen.future_force_mse_N2, mse.ravel(), rtol=1e-8, atol=1e-13)
                        raw_checks += len(chosen)
    physical_table = pd.DataFrame(physical)
    physical_table.to_csv(ROOT / "pilot_reference_response_spread.csv", index=False)
    summary = pd.read_csv(DATA / "summary.csv")
    selected = summary[(summary.duration_s.isin([0., .025, .1, .5, 1.]))
                       & (summary.sigma_N == .2)][["world", "probe_kind", "duration_s", "condition",
                                                  "future_force_rmse_N", "current_state_rmse"]]
    selected.to_csv(ROOT / "pilot_selected_results.csv", index=False)
    result = {"passed": True, "source_and_output_hashes_passed": True,
              "sensed_rows_independently_recomputed": raw_checks,
              "all_rows_finite": len(table), "complete_run_seconds": run["seconds"],
              "physical_response_spread_rows": len(physical),
              "scope": "Descriptive material pilot, fixed r1 clean models, no independent model retraining; no robot/controller result",
              "run_manifest_sha256": sha(DATA / "run.json")}
    (ROOT / "pilot_independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()

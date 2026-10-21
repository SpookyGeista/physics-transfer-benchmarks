"""Bounded CPU-only material-memory mechanism experiment; no new weight fits."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/pilot-memory-numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pilot-memory-mpl")
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from observer import candidate_bank, posterior_weights, summarize_posterior
from reference_fixture import run_fixture

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def commands(cfg):
    dt = cfg["dt"]
    steps = np.rint(np.array(cfg["durations_s"]) / dt).astype(int)
    pv = np.full(steps.max(), cfg["probe_velocity_m_s"])
    pn = np.full(steps.max(), cfg["probe_normal_N"])
    t = np.arange(round(cfg["future_seconds"] / dt)) * dt
    fv = .04 * np.cos(2 * np.pi * t / cfg["future_seconds"])
    fn = 15. + 3. * np.sin(2 * np.pi * t / cfg["future_seconds"] + .3)
    return steps, pv, pn, fv, fn


def figure(summary, physical, path, probe_kind):
    summary = summary[summary.probe_kind == probe_kind]
    physical = physical[physical.probe_kind == probe_kind]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
    metrics = [("initial_state_nrmse", "Alter Anfangszustand: NRMSE"),
               ("current_state_nrmse", "Zustand nach Sonde: NRMSE"),
               ("future_force_rmse_N", "Kraft-Forecast, nächste 0,15 s: RMSE [N]")]
    for row, world in enumerate(("maxwell", "boucwen")):
        for col, (metric, title) in enumerate(metrics):
            ax = axes[row, col]
            blind = summary[(summary.world == world) & (summary.condition == "blinded")
                            & (summary.sigma_N == summary.sigma_N.min())]
            ax.plot(blind.duration_s, blind[metric], "k--o", lw=1.8, ms=3,
                    label="Kraft ausgeblendet; gleiche Sonde")
            for sigma, color in zip(sorted(summary.sigma_N.unique()), ["#2166ac", "#d95f02"]):
                q = summary[(summary.world == world) & (summary.condition == "sensed")
                            & (summary.sigma_N == sigma)]
                ax.plot(q.duration_s, q[metric], "o-", color=color, ms=3,
                        label=f"Kraft beobachtet, σ={sigma:g} N")
            ax.set_xscale("symlog", linthresh=.005)
            ax.set_yscale("log")
            ax.set_xlabel("Dauer derselben Probe [s]")
            ax.set_title(f"{world}: {title}", fontsize=10)
            ax.grid(alpha=.22, which="both")
            if col == 1:
                ratio = physical[(physical.world == world) & (physical.duration_s == 1)].iloc[0].physical_spread_ratio
                ax.text(.03, .06, f"Physische Spreizung nach 1 s:\n{ratio:.2g} × Anfangsspreizung",
                        transform=ax.transAxes, fontsize=8)
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle(f"{probe_kind.upper()}: Information und physische Rekonditionierung trennen", fontsize=14)
    fig.supxlabel("Privilegierter Materialprüfstand: vorgegebenes v/N · eingefrorene State-ID · keine Robotik-/MPC-Bewertung\n"
                  "NRMSE normiert durch den ursprünglichen Priorbereich; Kurven sind deskriptive Ensemble-/Rauschmittel, keine Konfidenzintervalle",
                  fontsize=9)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        parser.error("Outputs must stay within this new pilot directory")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Existing results retained; choose a fresh --out")
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "config.json").read_text())
    models = {w: REPO / f"research_next/results/round1/models/state_id/{w}/r1/clean/blind_selection/model.json"
              for w in cfg["worlds"]}
    external = [REPO / "study_v2/src/decision_study/reference.py",
                REPO / "research_next/src/state_id.py", *models.values()]
    sources = {str(p): sha(p) for p in [*ROOT.glob("*.py"), ROOT / "config.json", ROOT / "README.md", *external]}
    write_json(out / "design_lock.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
               sources_sha256=sources, role="mechanism pilot declaration before observations/results; not statistical preregistration"))
    started = time.perf_counter()
    steps, slip_v, pn, fv, fn = commands(cfg)
    records, physical = [], []
    for world, probe_kind in ((w, p) for w in cfg["worlds"] for p in cfg["probe_kinds"]):
        wi = cfg["worlds"].index(world)
        pv = np.zeros_like(slip_v) if probe_kind == "rest" else slip_v.copy()
        if world == "maxwell":
            initial = np.array(cfg["maxwell_true_values"])
            grid = np.linspace(*cfg["maxwell_grid_bounds"], cfg["grid_points"])
        else:
            initial = np.geomspace(*cfg["boucwen_truth_geometric_bounds_s"], cfg["truth_count"])
            grid = np.geomspace(*cfg["boucwen_grid_bounds_s"], cfg["grid_points"])
        # The bank and observer receive commands and their prior, never truth.
        bank = candidate_bank(models[world], grid, pv, pn, fv, fn, steps, cfg["delta_m"], cfg["dt"])
        truth = run_fixture(world, initial, pv, pn, fv, fn, steps, cfg["delta_m"], cfg["dt"])
        assert np.all(truth["at_rest_force"] == 0.)
        assert np.max(truth["force"] * pv[None]) <= 1e-12
        assert np.max(truth["future"] * fv[None, None]) <= 1e-12
        spread = np.ptp(truth["current"], axis=1) / np.ptp(initial)
        np.testing.assert_allclose(spread, truth["analytic_contraction"], rtol=2e-8, atol=1e-13)
        noise = np.random.default_rng(cfg["noise_seed"] + wi).normal(
            size=(len(cfg["force_noise_sigma_N"]), len(initial), cfg["noise_replicates"], len(pv)))
        observed = truth["force"][None, :, None, :] + np.asarray(cfg["force_noise_sigma_N"])[:, None, None, None] * noise
        posteriors = []
        for di, (duration, count) in enumerate(zip(cfg["durations_s"], steps)):
            if time.perf_counter() - started > cfg["compute_budget_seconds"]:
                raise TimeoutError("Declared CPU budget reached; retain partial files")
            energy = -cfg["dt"] * np.sum(truth["force"][:, :count] * pv[None, :count], axis=1)
            physical.append(dict(world=world, probe_kind=probe_kind, duration_s=duration, samples=int(count),
                physical_spread_ratio=spread[di], analytic_spread_ratio=truth["analytic_contraction"][di],
                initial_spread=float(np.ptp(initial)), current_spread=float(np.ptp(truth["current"][di])),
                slip_distance_m=float(np.sum(np.abs(pv[:count])) * cfg["dt"]),
                mean_dissipated_probe_energy_J=float(energy.mean()),
                fixture_scope="imposed kinematics and normal load; energy omits actuators/clamp/inertia"))
            sigma_posteriors = []
            for si, sigma in enumerate(cfg["force_noise_sigma_N"]):
                y = observed[si, :, :, :count].reshape(len(initial) * cfg["noise_replicates"], count)
                conditioned = posterior_weights(bank["force"][:, :count], y, sigma)
                blind = posterior_weights(bank["force"][:, :count], None, sigma, batch_size=len(y))
                if count == 0 or probe_kind == "rest":
                    np.testing.assert_array_equal(conditioned, blind)
                sigma_posteriors.append(conditioned.reshape(len(initial), cfg["noise_replicates"], len(grid)))
                for condition, weights in [("blinded", blind), ("sensed", conditioned)]:
                    result = summarize_posterior(weights, grid, bank["current"][di], bank["future"][di])
                    variance = np.maximum(weights @ (bank["future"][di]**2) - result["future_mean"]**2, 0.)
                    for ti, true_initial in enumerate(initial):
                        for repeat in range(cfg["noise_replicates"]):
                            k = ti * cfg["noise_replicates"] + repeat
                            err = result["future_mean"][k] - truth["future"][di, ti]
                            records.append(dict(world=world, probe_kind=probe_kind, duration_s=duration, probe_samples=int(count),
                                sigma_N=sigma, initial_case=ti, noise_repeat=repeat, condition=condition,
                                physical_run_id=f"{world}/{probe_kind}/initial{ti}/steps{count}",
                                true_initial=float(true_initial), estimated_initial=result["initial_mean"][k],
                                true_current=truth["current"][di, ti], estimated_current=result["current_mean"][k],
                                initial_squared_error=float((result["initial_mean"][k] - true_initial)**2),
                                current_squared_error=float((result["current_mean"][k] - truth["current"][di, ti])**2),
                                initial_prior_span=float(np.ptp(grid)), future_force_mse_N2=float(np.mean(err**2)),
                                future_posterior_sd_rms_N=float(np.sqrt(variance[k].mean())),
                                initial_width90=result["initial_width90"][k], current_width90=result["current_width90"][k],
                                initial_entropy_nats=result["entropy"][k],
                                probe_slip_m=physical[-1]["slip_distance_m"], probe_energy_J=energy[ti]))
            posteriors.append(np.stack(sigma_posteriors))
        np.savez_compressed(out / f"{world}_{probe_kind}_raw.npz", initial_grid=grid, true_initial=initial,
            probe_v=pv, probe_normal=pn, future_v=fv, future_normal=fn,
            true_probe_force=truth["force"], measured_probe_force=observed,
            true_current=truth["current"], true_future_force=truth["future"],
            candidate_probe_force=bank["force"], candidate_current=bank["current"],
            candidate_future_force=bank["future"], sensed_posterior=np.stack(posteriors),
            durations=np.asarray(cfg["durations_s"]), force_sigmas=np.asarray(cfg["force_noise_sigma_N"]))
    frame = pd.DataFrame(records)
    expected = 2 * len(cfg["probe_kinds"]) * cfg["truth_count"] * len(steps) * len(cfg["force_noise_sigma_N"]) * cfg["noise_replicates"] * 2
    assert len(frame) == expected == 25920
    identity = ["world", "probe_kind", "duration_s", "sigma_N", "initial_case", "noise_repeat"]
    assert not frame.duplicated(identity + ["condition"]).any()
    pairs = frame.set_index(identity).groupby(level=list(range(len(identity))))
    for _, pair in pairs:
        assert len(pair) == 2
        for key in ["physical_run_id", "true_initial", "true_current", "probe_slip_m", "probe_energy_J"]:
            assert pair[key].nunique() == 1
    blind = frame[frame.condition == "blinded"]
    for _, group in blind.groupby(["world", "probe_kind", "sigma_N", "initial_case", "noise_repeat"]):
        np.testing.assert_allclose(group.estimated_initial, group.estimated_initial.iloc[0], atol=1e-14)
        np.testing.assert_allclose(group.initial_entropy_nats, np.log(cfg["grid_points"]), atol=1e-13)
    summary = frame.groupby(["world", "probe_kind", "duration_s", "sigma_N", "condition"], sort=True).agg(
        cases=("true_initial", "size"), initial_mse=("initial_squared_error", "mean"),
        current_mse=("current_squared_error", "mean"), future_mse=("future_force_mse_N2", "mean"),
        prior_span=("initial_prior_span", "first"), mean_initial_width90=("initial_width90", "mean"),
        mean_current_width90=("current_width90", "mean"), mean_initial_entropy=("initial_entropy_nats", "mean"),
        mean_future_posterior_sd_rms_N=("future_posterior_sd_rms_N", "mean")).reset_index()
    summary["initial_state_rmse"] = np.sqrt(summary.initial_mse)
    summary["current_state_rmse"] = np.sqrt(summary.current_mse)
    summary["initial_state_nrmse"] = summary.initial_state_rmse / summary.prior_span
    summary["current_state_nrmse"] = summary.current_state_rmse / summary.prior_span
    summary["future_force_rmse_N"] = np.sqrt(summary.future_mse)
    physical = pd.DataFrame(physical)
    frame.to_csv(out / "trials.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    physical.to_csv(out / "physical_contraction.csv", index=False)
    for probe_kind in cfg["probe_kinds"]:
        figure(summary, physical, out / f"memory_mechanism_{probe_kind}.png", probe_kind)
    assert sources == {p: sha(p) for p in sources}, "Imported/model/pilot source changed"
    seconds = time.perf_counter() - started
    if seconds > cfg["compute_budget_seconds"]:
        raise TimeoutError("Pilot exceeded declared CPU budget; retain results as incomplete")
    write_json(out / "run.json", dict(status="complete", seconds=seconds, rows=len(frame),
        summary_rows=len(summary), physical_rows=len(physical), device="cpu", weight_updates=0,
        models={k: str(v) for k, v in models.items()}, source_sha256=sources,
        outputs_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()},
        checks=dict(zero_duration_no_information=True, resting_force_no_information=True, blinded_prior_unchanged=True,
            identical_physical_counterfactuals=True, exact_analytic_contraction=True,
            passive_tangential_power=True, source_hashes_unchanged=True),
        uncertainty_scope="fixed-grid conditional posterior, known noise; no model-family/parameter uncertainty",
        statement="Exploratory constitutive mechanism pilot; no statistical confirmation, no robotics or novelty claim"))
    print(json.dumps(dict(status="complete", seconds=seconds, rows=len(frame), output=str(out)), indent=2))


if __name__ == "__main__":
    main()

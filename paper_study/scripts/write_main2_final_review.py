#!/usr/bin/env python3
"""Post-lock synthesis of completed, independently audited Main2 outputs.

The objective-gain magnitude summary is explicitly supplementary descriptive
analysis; it changes no optimizer tie rule or planned contrast.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "paper_study/results/statistics_main2"
R = ROOT / "paper_study/reports"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def table(columns, rows):
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])

def main():
    out_md = R / "main2_final_review.md"
    out_json = R / "main2_final_review.json"
    if out_md.exists() or out_json.exists():
        raise FileExistsError("Review is final; do not overwrite")
    manifest = json.loads((S / "artifact_manifest.json").read_text())
    for path, expected in manifest["outputs_sha256"].items():
        assert sha(path) == expected, path
    lock = json.loads((S / "analysis_lock.json").read_text())
    for path, expected in lock["sources_sha256"].items():
        assert sha(path) == expected, path
    summary = json.loads((S / "summary.json").read_text())
    process = json.loads((S / "process.json").read_text())
    assert process["status"] == "complete"
    pooled = pd.read_csv(S / "pooled_losses.csv")
    contrasts = pd.read_csv(S / "contrasts.csv")
    nested = pd.read_csv(S / "nested_mle.csv")
    costs = pd.read_csv(S / "native_costs.csv")
    hash_checks = pd.read_csv(S / "hash_checks.csv")
    positive = nested.loc[nested.past_improvement > 0]
    quantiles = positive.past_improvement.quantile([0, .25, .5, .75, .9, .99, 1])
    near_equal = []
    for threshold in [1e-12, 1e-8, 1e-4, .01]:
        cell = positive[positive.past_improvement <= threshold]
        near_equal.append({"threshold": threshold, "selected_count": len(cell), "future_position_worse": int((cell.future_mse_change_position_m > 0).sum())})
    methods = ["prior_propagation", "trajectory_assimilation", "privileged_full_history", "bounded_mle_start1", "bounded_mle_two_start"]
    labels = ["Prior propagation", "Trajectory assimilation (P1024)", "Privileged fitted history", "Bounded MLE: start 1", "Bounded MLE: selected two-start"]
    global_pooled = pooled[pooled.scope == "global"].set_index(["method", "channel"])
    global_table = table(["Method", "Position RMSE (mm)", "Velocity RMSE (mm/s)", "Normal RMSE (N)", "Friction RMSE (N)"], [
        [label] + [f"{global_pooled.loc[(method, channel), 'pooled_rmse'] * scale:.6g}" for channel, scale in [("position_m", 1000), ("velocity_m_s", 1000), ("normal_N", 1), ("friction_N", 1)]]
        for method, label in zip(methods, labels)])
    probe_ratios = contrasts[(contrasts.scope == "probe") & (contrasts.channel == "position_m")].set_index(["probe", "contrast"])
    ratio_table = table(["Preparation", "Assimilation / prior MSE", "MLE1 / assimilation MSE", "MLE2 / MLE1 MSE"], [
        [probe] + [f"{probe_ratios.loc[(probe, contrast), 'mse_ratio']:.6f}" for contrast in ["assimilation/prior", "MLE1/assimilation", "MLE2/MLE1"]]
        for probe in ["hold_150", "gentle_150", "strong_300"]])
    costs["future_two_port_absolute_J"] = costs.total_two_port_absolute_J - costs.probe_two_port_absolute_J
    grouped_costs = costs.groupby("probe")[["probe_seconds", "total_seconds", "probe_two_port_absolute_J", "future_two_port_absolute_J", "total_two_port_absolute_J"]].mean()
    cost_table = table(["Preparation", "Probe / total duration (s)", "Probe work (J)", "Future work (J)", "Total work (J)"], [
        [p, f"{row.probe_seconds:g} / {row.total_seconds:g}"] + [f"{row[k]:.6f}" for k in ["probe_two_port_absolute_J", "future_two_port_absolute_J", "total_two_port_absolute_J"]]
        for p, row in grouped_costs.iterrows()])
    text = f"""# Main2 final independent review

**The complete audit passed. The descriptive result is mixed, with no general benefit from more likelihood optimization.** Across the full design, trajectory assimilation reduces future position MSE to 0.032466 of prior propagation. The bounded one-start estimate has 1.294960 times the assimilation MSE; the selected two-start estimate has 2.056405 times the one-start MSE. These are ratios of pooled episode MSE, not averages of episode ratios and not significance tests.

## Complete design, provenance and numerical checks

The audit covers all 96 native batches (3,456 physical episodes), 288 forecast batches across three previously fitted banks, 10,368 bank/episode contexts and 51,840 method rows. Four held-out history blocks contain six C2 histories each, crossed with eight material compositions, three known geometries, two excitation signs and three preparations. The five methods, poor predictions, optimizer budget cases and coincident forecasts are all retained. The independent analysis source, tests and plan were locally locked before analyst inspection of Main2 losses; this is not external preregistration.

The completed analysis performed {process['hash_checks']:,} source/model/prior/authorization/input/output hash-binding checks before and after execution. Its original 22 output hashes and four analysis-source hashes were verified again for this review. Identities and exact factorial inventory were checked across native NPZ, metadata, forecast arrays and CSV. All four-channel episode MSE, RMSE and signed bias were independently recalculated from saved predictions and native targets; maximum difference was {summary['max_score_recompute_absolute_difference']:.3g}. The two optimizer candidate forecasts, strict past-objective selection, tie-to-start-1 rule, causal best-particle initialization, normalized posterior weights, ESS, clocks and recorded solver budgets were also checked.

Native guards total zero. Maximum Newton residual was {summary['max_native_newton_residual_N']:.3g} N and kinematic residual {summary['max_native_kinematic_residual_m']:.3g} m. Recorded implicit material-solver residuals satisfy the fixed 10⁻⁷ N bound. Individual future-particle guard trajectories are not stored, so the audit does not assert their absence. The full independent recomputation took {process['seconds']:.2f} s; no simulator or fitting was rerun.

## All-method forecast results and exceptions

Each value below is the square root of the equally weighted mean episode MSE; mean episode RMSE is separately stored in the complete tables.

{global_table}

The aggregate deterioration is not universal. The strong probe has lower MSE for MLE1 than assimilation, and slightly lower MSE for MLE2 than MLE1:

{ratio_table}

MLE2 exceeds MLE1 in the pooled position MSE of each of the eight compositions, each bank and each history block. MLE1 improves on assimilation for the Maxwell/balanced and Maxwell/compliant compositions; the other six composition aggregates worsen. These are complete prespecified descriptive breakdowns, not independent significance results. All channels and groups, absolute differences and ratios are in [pooled losses](../results/statistics_main2/pooled_losses.csv) and [contrasts](../results/statistics_main2/contrasts.csv).

## Near-equal past fits and different future predictions

The second fixed start was selected in {len(positive):,} of {len(nested):,} contexts. Future position MSE increased in 3,640, decreased in 1,701, and remained unchanged in 5,027. The mean past squared-normalized-residual objective decreased from {nested.past_start1.mean():.6f} to {nested.past_selected.mean():.6f}.

**Supplementary post-lock descriptive qualification:** most strict gains were extremely small. Among positive gains, the median was {quantiles.loc[.5]:.3g}, the 75th percentile {quantiles.loc[.75]:.3g}, and the 90th percentile {quantiles.loc[.9]:.6f}. In 4,084 of the 5,341 selections the gain was at most 10⁻⁸ objective units; 2,822 of those forecasts had larger position MSE. Thus the result is better described as *near-equal past likelihoods sometimes producing very different future predictions*, together with a minority of larger objective gains. It does not establish materially better past fit in every selected case. These magnitude summaries introduce no new acceptance threshold or tie-break: the frozen strict floating-point selection and every original score remain unchanged.

All 20,736 optimizer starts are retained, including six function-evaluation-budget/nonconverged cases (120 evaluations). The continuous three-dimensional MLE box has wider support than the empirical particle-history prior; the methods therefore differ in structural support as well as computation. P1024 did not pass development convergence checks. Posterior ESS median {summary['posterior_ess']['median']:.3f}, minimum 1 and maximum {summary['posterior_ess']['max']:.3f} show concentration, not calibrated uncertainty or physically correct latent-state identification.

## Physical work and measured computation

Physical work is counted once per native episode, independent of the three model banks and five estimators. The discrete two-port absolute work is Σ|u Δx| + Σ|N Δd|; actual native displacement is used. Probe and subsequent 400 ms forecast-program work are separate:

{cost_table}

The external ideal 4 s C2 preparation fixture is outside this native-work accounting and is not a force-limited robot. Signed components, friction work and the distinct normal-drive ZOH quantity N d_dot dt remain in [native_costs.csv](../results/statistics_main2/native_costs.csv).

Summed measured batch wall time is {summary['compute']['particle_total_seconds']:.2f} s for particle assimilation and prior/posterior forecasts, plus {summary['compute']['mle_incremental_seconds']:.2f} s for the two MLE starts and their forecasts. MLE includes its required particle initialization: {summary['compute']['mle_including_particle_seconds']:.2f} s combined. Privileged fitted-history replay took {summary['compute']['privileged_seconds']:.2f} s. Summed bank process times were {summary['compute']['summed_bank_current_attempt_seconds']:.2f} s; the banks ran in parallel. These are measured batch wall times, not CPU-seconds or single-instance real-time claims. The standalone cost of each individual MLE variant was not separately measured.

## Information rights and scope

Regular methods receive only causal noisy x/v/N samples, known geometry and commands, including boundary N after the known future-rate invocation. This is a conditional prediction comparison, not an early stopping or control-improvement claim. Prior propagation shares the sensor-dependent known commands and is a no-likelihood algorithmic ablation, not an information-theoretically blind observer. Constitutive weights stay fixed. Privileged fitted-history replay receives the true preparation history but not true parameters; under model mismatch it is not an oracle or a universal error floor.

The independent units are three existing calibration-bank pairs and four held-out history blocks, not 10,368 independent trials or 51,840 independent method observations. No confidence intervals, p-values, optimizer-global-optimum, posterior-convergence, NN novelty or broad material-population claims follow. The known h/h4 future-trajectory drift of 0.285–0.881 mm limits interpretation to the fixed 2.5 ms simulation; matched-timestep model agreement does not establish continuous-physics accuracy. Main2 is a descriptive extension and does not replace Main1's confirmatory contrasts. The real G1 remains unused.

## Figures, reproducibility and review artifacts

The original four German PNG/PDF figures were inspected and remain unchanged in [statistics_main2/figures](../results/statistics_main2/figures). The new [English PNG](figures/main2_final/main2_past_vs_future_en.png) and [vector PDF](figures/main2_final/main2_past_vs_future_en.pdf) retain every context, all three probes and banks, shared axes and x ≥ 0. The 7 × 3.35 inch layout has readable labels, no cropped content and no raster images in the PDF. Input/source/output hashes and point counts are in [figure provenance](figures/main2_final/provenance.json); [caption](figures/main2_final/caption.md) documents transforms and overlap. This presentation-only export and the magnitude qualification above were added after the analysis lock.

[Frozen protocol](../costed_benchmark/main2_protocol.md), [analysis plan](main2_analysis_plan.md), [analysis source](../scripts/analyze_main2.py), [full report](../results/statistics_main2/REPORT.md), [analysis lock](../results/statistics_main2/analysis_lock.json), [artifact manifest](../results/statistics_main2/artifact_manifest.json), [raw forecast audit](../results/statistics_main2/forecast_audit.csv), [optimizer starts](../results/statistics_main2/optimizer_starts.csv), [nested selections](../results/statistics_main2/nested_mle.csv), [machine-readable final review](main2_final_review.json).

Re-export the presentation to a fresh directory with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/loya2/miniconda3/envs/unitree_sim_env/bin/python paper_study/scripts/plot_main2_presentation.py --out /tmp/main2_presentation_reproduction`. The script verifies all original output hashes and refuses to overwrite existing exports.
"""
    out_md.write_text(text)
    review = {
        "status": "complete_independent_audit_passed",
        "scope": "Descriptive held-out Main2; three fitted banks and four history blocks; no inferential pass/fail",
        "analysis_summary": summary,
        "analysis_process": process,
        "analysis_hash_bindings": len(hash_checks),
        "unique_hash_paths": hash_checks.path.nunique(),
        "original_output_hashes_reverified": len(manifest["outputs_sha256"]),
        "post_lock_objective_magnitude_description": {
            "changes_any_selection_or_gate": False,
            "positive_gain_quantiles": {str(k): float(v) for k, v in quantiles.items()},
            "mean_start1_objective": float(nested.past_start1.mean()),
            "mean_selected_objective": float(nested.past_selected.mean()),
            "descriptive_threshold_counts": near_equal,
        },
        "probe_position_ratios": contrasts.loc[(contrasts.scope == "probe") & (contrasts.channel == "position_m"), ["probe", "contrast", "mse_ratio", "absolute_mse_difference"]].to_dict("records"),
        "costs_counted_once_per_physical_episode": grouped_costs.reset_index().to_dict("records"),
        "figure_audit": {"all_original_figures_visually_checked": True, "english_export_visually_checked": True, "english_pdf_page_size_points": [504, 241.2], "english_pdf_raster_image_count": 0},
        "provenance_sha256": {str(p): sha(p) for p in [Path(__file__), out_md, S / "artifact_manifest.json", S / "analysis_lock.json", R / "figures/main2_final/provenance.json"]},
    }
    out_json.write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(out_md), "report_sha256": sha(out_md), "json": str(out_json), "json_sha256": sha(out_json)}, indent=2))

if __name__ == "__main__":
    main()

"""Predeclared crossed-block bootstrap for the held-out Main1 replication.

Eight fitted model banks and eight independent history blocks are the resampled
units. Native timesteps, signs, cam settings and repeated estimators are never
counted as independent observations. See reports/main1_protocol.md.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
COMPONENTS=[("boucwen","gentle_150","trajectory_assimilation"),
            ("boucwen","gentle_150","prior_propagation"),
            ("maxwell","gentle_150","trajectory_assimilation"),
            ("maxwell","gentle_150","prior_propagation"),
            ("boucwen","hold_150","trajectory_assimilation")]
NAMES=["aging_assimilation_ratio","aging_minus_relaxing_relative_gain","aging_probe_vs_hold_ratio"]


def contrasts(means):
    """Ratio of equal-weight pooled MSEs, NOT average of per-episode ratios."""
    means=np.asarray(means,dtype=float)
    if means.shape[-1]!=5 or not np.isfinite(means).all() or np.any(means[..., [1,3,4]]<=0):
        raise ValueError("Need five finite losses with positive comparison denominators")
    aging=means[...,0]/means[...,1]
    relaxing=means[...,2]/means[...,3]
    return np.stack((aging,relaxing-aging,means[...,0]/means[...,4]),axis=-1)


def loss_cube(frame, *, main=False):
    if main:
        expected_categories={"sensor":{"noisy"},"calibration_noise":{"noisy"},
            "forecast_start":{"filtered_particles"},"filter_kind":{"joint_sensor_likelihood"},
            "normal_world":{"maxwell","boucwen"},"friction_world":{"maxwell","boucwen"},
            "estimator":{"prior_propagation","trajectory_assimilation","privileged_full_history"}}
        for key,expected in expected_categories.items():
            if set(frame[key].unique())!=expected:
                raise ValueError(f"Unexpected or missing Main1 categories: {key}")
        if "method" in frame and not (frame.method==frame.filter_kind+"/"+frame.forecast_start+"/"+frame.estimator).all():
            raise ValueError("Main1 redundant method labels disagree")
    d=frame[(frame.sensor=="noisy")&(frame.calibration_noise=="noisy") &
        (frame.forecast_start=="filtered_particles") &
        (frame.filter_kind=="joint_sensor_likelihood")].copy()
    if d.empty: raise ValueError("No eligible primary rows")
    identity=["model_seed","preparation_seed","normal_world","friction_world","probe","episode","estimator"]
    if d.duplicated(identity).any(): raise ValueError("Duplicate primary episode keys")
    if not np.isfinite(d.future_position_rmse_m).all() or (d.future_position_rmse_m<0).any():
        raise ValueError("RMSE must be finite and nonnegative; invalid losses cannot be silently removed")
    if main:
        if sorted(d.model_seed.unique().tolist())!=list(range(1,9)):
            raise ValueError("Main1 requires all eight declared model banks")
        if sorted(d.preparation_seed.unique().tolist())!=list(range(51201,51209)):
            raise ValueError("Main1 requires all eight reserved history blocks")
        if sorted(d.probe.unique().tolist())!=sorted(["none","hold_050","hold_150","hold_300","gentle_050","gentle_150","strong_150","strong_300"]):
            raise ValueError("Main1 probe set incomplete")
        sizes=d.groupby(["model_seed","preparation_seed","normal_world","friction_world","probe","estimator"]).size()
        if len(sizes)!=8*8*2*2*8*3 or not (sizes==48).all():
            raise ValueError("Main1 factorial cells incomplete")
        if any(set(group.episode)!=set(range(48)) for _,group in d.groupby(
                ["model_seed","preparation_seed","normal_world","friction_world","probe","estimator"])):
            raise ValueError("Main1 episode IDs incomplete")
    d["mse"]=d.future_position_rmse_m**2
    banks=np.sort(d.model_seed.unique()); histories=np.sort(d.preparation_seed.unique())
    all_pairs=pd.MultiIndex.from_product([banks,histories],names=["model_seed","preparation_seed"])
    cube=np.empty((len(banks),len(histories),len(COMPONENTS)))
    for j,(friction,probe,estimator) in enumerate(COMPONENTS):
        cell=d[(d.friction_world==friction)&(d.probe==probe)&(d.estimator==estimator)]
        grouped=cell.groupby(["model_seed","preparation_seed"]).mse.mean().reindex(all_pairs)
        counts=cell.groupby(["model_seed","preparation_seed"]).size().reindex(all_pairs)
        if grouped.isna().any() or counts.nunique()!=1:
            raise ValueError("Missing or unequally weighted primary blocks")
        cube[...,j]=grouped.to_numpy().reshape(len(banks),len(histories))
    return cube,banks,histories


def crossed_bootstrap(cube, *, repetitions=10000, seed=91551201):
    cube=np.asarray(cube,dtype=float)
    if cube.ndim!=3 or cube.shape[-1]!=5 or not np.isfinite(cube).all():
        raise ValueError("Expected finite bank x history x five-loss array")
    banks,histories,_=cube.shape
    if min(banks,histories)<2: raise ValueError("Development singleton blocks do not support this CI")
    rng=np.random.default_rng(seed)
    sampled_banks=rng.integers(banks,size=(repetitions,banks))
    sampled_histories=rng.integers(histories,size=(repetitions,histories))
    bw=np.eye(banks)[sampled_banks].sum(axis=1)/banks
    hw=np.eye(histories)[sampled_histories].sum(axis=1)/histories
    means=np.einsum("bm,bh,mhc->bc",bw,hw,cube)
    values=contrasts(means)
    interval=np.quantile(values,[.005,.995],axis=0)
    return dict(point=contrasts(cube.mean(axis=(0,1))),interval=interval,
                bootstrap_values=values, pooled_mse=cube.mean(axis=(0,1)))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_evaluation_artifact(data):
    """Verify the scored CSV against the evaluator's completed output manifest."""
    data=Path(data).resolve()
    manifest=json.loads((data/"artifact_manifest.json").read_text())
    if sha(data/"evaluation_lock.json")!=manifest["evaluation_lock_sha256"]:
        raise ValueError("Evaluation lock differs from its output manifest")
    expected=manifest["outputs_sha256"].get(str(data/"episodes.csv"))
    if expected is None or sha(data/"episodes.csv")!=expected:
        raise ValueError("Episode CSV differs from the evaluator's completed output manifest")
    return sha(data/"artifact_manifest.json")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to(ROOT/"results") or out==ROOT/"results" or out.exists() and any(out.iterdir()):
        raise ValueError("Use fresh results subdirectory")
    process=json.loads((args.data/"process.json").read_text())
    if process["status"]!="complete":raise ValueError("Evaluation must be complete")
    artifact_manifest_sha256=verify_evaluation_artifact(args.data)
    lock=json.loads((args.data/"evaluation_lock.json").read_text())
    for path,digest in lock["sources_sha256"].items():
        if sha(path)!=digest:raise ValueError(f"Evaluation source hash mismatch: {path}")
    frame=pd.read_csv(args.data/"episodes.csv")
    cube,banks,histories=loss_cube(frame,main=True)
    out.mkdir(parents=True)
    provenance=dict(role="Analysis of locked Main1 contrasts",script_sha256=sha(__file__),
        protocol_sha256=sha(ROOT/"reports/main1_protocol.md"),input_csv_sha256=sha(args.data/"episodes.csv"),
        input_lock_sha256=sha(args.data/"evaluation_lock.json"),banks=banks.tolist(),history_blocks=histories.tolist(),
        artifact_manifest_sha256=artifact_manifest_sha256,
        bootstrap_seed=91551201,bootstrap_replicates=10000,ci_level=.99,
        caveat="Finite crossed-block percentile bootstrap; approximate uncertainty in the declared model/history population, not a review or general-material guarantee")
    (out/"analysis_lock.json").write_text(json.dumps(provenance,indent=2)+"\n")
    result=crossed_bootstrap(cube)
    intervals=result["interval"]
    passed=[bool(intervals[1,0]<.90),bool(intervals[0,1]>0),bool(intervals[1,2]<.90)]
    rows=[]
    for i,name in enumerate(NAMES):
        rows.append(dict(contrast=name,point=float(result["point"][i]),lower_99=float(intervals[0,i]),
                         upper_99=float(intervals[1,i]),criterion_passed=passed[i]))
    components=[]
    for c,mse in zip(COMPONENTS,result["pooled_mse"]):
        components.append(dict(friction=c[0],probe=c[1],estimator=c[2],mse_m2=float(mse),
                               rms_over_all_samples_mm=float(np.sqrt(mse)*1000)))
    mse=result["pooled_mse"]
    absolute_differences=dict(aging_assimilation_minus_prior_m2=float(mse[0]-mse[1]),
                              aging_probe_minus_hold_m2=float(mse[0]-mse[4]))
    summary=dict(contrasts=rows,components=components,absolute_mse_differences=absolute_differences,all_three=all(passed),
                 native_guard_rows=int((frame.native_guard_count>0).sum()),**provenance)
    (out/"results.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(out/"bootstrap.npz",loss_cube=cube,values=result["bootstrap_values"],
                         banks=banks,histories=histories)
    pd.DataFrame(rows).to_csv(out/"contrasts.csv",index=False)
    lines=["# Main1: vorab festgelegte zurückgehaltene Kontraste", "",
           f"Alle drei geplanten Kriterien erfüllt: **{all(passed)}**. Acht Modellbanks und acht Historienblöcke; 99%-Perzentilintervalle aus10.000 gekreuzten Bootstrap-Ziehungen. Keine unabhängige Wertung der einzelnen Zeitschritte oder Wiederholungen.","",
           "| Kontrast | Wert | Untere99% | Obere99% | Kriterium erfüllt |", "|---|---:|---:|---:|---:|"]
    for r in rows:lines.append(f"| {r['contrast']} | {r['point']:.6g} | {r['lower_99']:.6g} | {r['upper_99']:.6g} | {r['criterion_passed']} |")
    lines += ["", f"Absolute MSE-Differenz Assimilation minus Prior: {absolute_differences['aging_assimilation_minus_prior_m2']:.8g} m². Absolute MSE-Differenz aktive Probe minus Halten: {absolute_differences['aging_probe_minus_hold_m2']:.8g} m². Negative Werte bedeuten einen kleineren Vorhersageverlust.", "", "Die beiden Quotienten vergleichen gepoolte mittlere quadratische Positionsfehler; ihr Erfolg verlangt eine obere Grenze unter 0,90. Der Regimevergleich verlangt eine untere Grenze über 0. Die vollständigen absoluten Verluste stehen in results.json. Ein kleinerer Vorhersageverlust beweist keine richtige physische Gedächtnisrekonstruktion; Zustandskorrektur kann bei festem Modellbias eine Ersatzfunktion übernehmen.","",
              "Diese lokal gesperrte Replikation untersucht synthetische Materialien und eine bekannte Nockenfamilie. Sie beansprucht weder eine neue Bayes-Methode noch eine Garantie auf Publikationsannahme. Die Intervalle sind bei achtBlöcken jeAchse approximativ; die festgelegten Materialfamilien bilden keine zufällige Stichprobe aller physikalischen Systeme."]
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"all_three":all(passed),"contrasts":rows},indent=2))


if __name__=="__main__":main()

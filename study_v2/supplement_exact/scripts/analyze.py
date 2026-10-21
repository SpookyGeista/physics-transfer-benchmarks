"""Reuse the original frozen statistics/decision functions; do not pool arrangement sets."""
import sys,json,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import pandas as pd
import numpy as np
from decision_study.common import ROOT,CFG,dump,digest
from decision_study.locking import verify_lock
verify_lock();SUP=ROOT/'supplement_exact'
lock=json.loads((SUP/'lock.json').read_text())
for file,sha in lock['files'].items():assert digest(file)==sha
spec=importlib.util.spec_from_file_location('frozen_report',ROOT/'scripts/report.py');report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
pred=pd.read_csv(SUP/'results/prediction_metrics.csv');control=pd.read_csv(ROOT/'results/control_metrics.csv');support=pd.read_csv(SUP/'results/support_metrics.csv');support_summary=json.loads((SUP/'results/support_summary.json').read_text())
assert len(pred)==19200 and len(support)==960 and len(control)==1600
for world in CFG['worlds']:
 for seed in CFG['seeds']:
  for noise in CFG['noise_conditions']:
   for method in CFG['methods']:
    cell=pred[(pred.world==world)&(pred.seed==seed)&(pred.noise==noise)&(pred.method==method)];assert len(cell)==96 and cell.id.nunique()==96
pm=['position_rmse','velocity_rmse','normal_rmse','friction_rmse','conditional_normal_rmse','conditional_friction_rmse','guard_steps'];cm=['cost','success','tracking_rmse','terminal_position_error']
assert np.isfinite(pred[pm].to_numpy(dtype=float)).all() and np.isfinite(control[cm].to_numpy(dtype=float)).all()
psd=pred.groupby(['world','seed','budget','noise','method'],as_index=False)[pm].mean();csd=control.groupby(['world','seed','noise','method'],as_index=False)[cm].mean()
ps=report.summarize(psd,['world','budget','noise','method'],pm)
pairs=pd.concat([report.paired(psd,['world','budget','noise'],pm,'prediction'),report.paired(csd,['world','noise'],cm,'control')],ignore_index=True)
native,sources=report.native_statistics(SUP/'results');decision=report.decide(pairs,support,support_summary,native,pred)
decision.update(original_result_retained='inconclusive due to two strict exact-match failures',supplement='single prospectively locked construction correction with four new cam amplitudes',controller_scope='unchanged separate original force-driven task; no new-cam controller experiment',lock_sha256=digest(SUP/'lock.json'),model_scores_seen_before_supplement_lock=False)
for name,data in [('prediction_seed_means',psd),('prediction_summary',ps),('paired_differences',pairs),('native_summary',native)]:data.to_csv(SUP/f'results/{name}.csv',index=False)
dump(SUP/'results/decision.json',decision);dump(SUP/'results/analysis_sources.json',dict(native_sources=sources,prediction_sha256=digest(SUP/'results/prediction_metrics.csv'),original_control_sha256=digest(ROOT/'results/control_metrics.csv'),original_report_code_sha256=digest(ROOT/'scripts/report.py')))
print('SUPPLEMENT_COMPLETE',decision['outcome'],flush=True)

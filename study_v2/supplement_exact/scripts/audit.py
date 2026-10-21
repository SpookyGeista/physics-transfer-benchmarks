"""Unchanged exact-match and nearest-neighbour checks on the one new arrangement set."""
import json,sys,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
import pandas as pd
from decision_study.common import ROOT,CFG,NI,FI,dump,digest
from decision_study.locking import verify_lock
verify_lock();SUP=ROOT/'supplement_exact';lock=json.loads((SUP/'lock.json').read_text())
for file,sha in lock['files'].items():assert digest(file)==sha
spec=importlib.util.spec_from_file_location('original_support',ROOT/'scripts/support.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
config=json.loads((SUP/'config.json').read_text());rows=[]
for world in CFG['worlds']:
 for seed in CFG['seeds']:
  train,tm,_=audit.load_data(ROOT/f'results/data/{world}/r{seed}','train');replay,rm,_=audit.load_data(SUP/f'results/data/{world}/r{seed}','replay');exact=audit.audit_exact(train,tm['rows'],replay,rm['rows'],config['cam_amplitudes'])
  sp=ROOT/f'results/support/{world}/r{seed}.npz';metadata=json.loads(sp.with_suffix('.json').read_text());assert digest(sp)==metadata['npz_sha256'];points=dict(np.load(sp));diagnostics={}
  for head,columns in [('normal',NI),('friction',FI)]:diagnostics[head]=audit.episode_support(replay,points[f'{head}_b24'],metadata['heads'][f'{head}_b24']['threshold'],columns)
  for i,episode in enumerate(rm['rows']):
   row=dict(world=world,seed=seed,budget=24,episode=episode['id'],amplitude=episode['amplitude'],profile=episode['profile'],**exact[i])
   for head,values in diagnostics.items():row.update({head+'_'+key:value for key,value in values[i].items()})
   rows.append(row)
  print('SUPPLEMENT_SUPPORT',world,seed,sum(r['exact_match'] for r in exact),'/',len(exact),flush=True)
pd.DataFrame(rows).to_csv(SUP/'results/support_metrics.csv',index=False);dump(SUP/'results/support_summary.json',dict(primary_all_exact_match=all(r['exact_match'] for r in rows),episodes=len(rows),exact_passes=sum(r['exact_match'] for r in rows),no_episodes_excluded=True,tolerances=audit.TOLERANCES,lock_sha256=digest(SUP/'lock.json')))

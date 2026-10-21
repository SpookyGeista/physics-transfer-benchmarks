"""Bounded synthetic CPU smoke. Not Isaac, not native or held-out evidence."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/decision-study-numba')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
import argparse
import time
import numpy as np
from paper_study.decision_study.common import HERE,BRANCHES,load_config,fresh_output,sha,write_json,verify_hashes
from paper_study.decision_study.generate import run_triplet,factors
from paper_study.decision_study.inference import (load_model,model_paths,load_prior,prior_path,observable_prefix,
    filter_prefix,forecast_branch,branch_values,select_policies)


class SyntheticRig:
    def __init__(self,n,dt):self.n=n;self.dt=dt;self.reset()
    def reset(self):self.xv=np.zeros((self.n,2))
    def state(self):return self.xv.copy()
    def advance(self,force):
        self.xv[:,1]+=np.asarray(force)*self.dt/2.
        self.xv[:,0]+=self.dt*self.xv[:,1]
        return self.state()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();cfg=load_config();out=fresh_output(args.out)
    sources=sorted(HERE.glob('*.py'))+[HERE/'config.json',HERE/'protocol.md',
        REPO/'paper_study/src/preparation.py',REPO/'paper_study/src/preparation_v2.py',
        REPO/'study_v2/src/decision_study/reference.py',REPO/'study_v2/configs/experiment.json',
        REPO/'paper_study/costed_benchmark/geometry.py',
        REPO/'research_next/src/state_id.py']
    for n,f in cfg['compositions']:sources.extend((*model_paths(n,f,cfg),prior_path(n,f,cfg)))
    hashes={str(p):sha(p) for p in sources};write_json(out/'source_lock.json',hashes)
    started=time.perf_counter();records=[]
    for nw,fw in cfg['compositions']:
        tick=time.perf_counter();rows=factors(34001,cfg)[:6]
        arrays,metadata=run_triplet(SyntheticRig(3*len(rows),cfg['dt']),nw,fw,34001,cfg,factor_rows=rows,
            deadline=started+120.)
        generation=time.perf_counter()-tick;tick=time.perf_counter()
        model=load_model(nw,fw,cfg);grid,prior=load_prior(nw,fw,cfg)
        public=observable_prefix(arrays['release'],cfg);snap=filter_prefix(model,grid,prior,public,cfg)
        branches={b:forecast_branch(model,snap,public,b,cfg) for b in BRANCHES}
        values=branch_values(snap,branches,cfg);policies=select_policies(values,cfg)
        record=dict(normal=nw,friction=fw,contexts=len(rows),synthetic_generation_seconds=generation,
            planning_seconds=time.perf_counter()-tick,prefix_bitwise=metadata['prefix_audit']['bitwise_identical'],
            risk_finite=bool(np.isfinite(values['risk_m2']).all()),
            candidate_guards=int(sum(v['candidate_guard_steps'].sum() for v in branches.values())),
            policy_count=len(policies['B0']['policies']),raw_candidate_future_bytes=sum(v['future'].nbytes for v in branches.values()))
        records.append(record);print(record,flush=True)
        if time.perf_counter()-started>120.:raise TimeoutError('CPU smoke exceeded120 seconds')
    verify_hashes(hashes)
    import resource
    write_json(out/'summary.json',dict(role='Synthetic float64 CPU smoke only; not native validation or scientific outcomes',
        status='complete',seconds=time.perf_counter()-started,cases=records,
        peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_lock_sha256=sha(out/'source_lock.json')))


if __name__=='__main__':main()

"""Separate ideal-sensor h/2 and h/4 PI-native refinement.

Both the external C2 fixture and native phase use the refined timestep here.
Every turning point remains on the canonical grid; rate-independent stop
preparation should therefore be subdivision invariant up to rounding. That is
checked from saved initial states, not assumed as a numerical test result.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/paper-study-numba')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse
import json
import time
from generate import ROOT,HERE,REPO,validate_config,source_paths,sha,write_json,run_batch,factorial_design


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--divisor',type=int,choices=[2,4],required=True)
    parser.add_argument('--out',type=Path,required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.device!='cpu':raise ValueError('CPU PhysX only')
    source=HERE/'development.json';cfg=json.loads(source.read_text());validate_config(cfg)
    cfg['dt']/=args.divisor;cfg['observation_steps']*=args.divisor
    cfg['sensor_conditions']={'ideal':cfg['sensor_conditions']['ideal']}
    cfg['probes']=[p for p in cfg['probes'] if p['name'] in ('hold_150','gentle_150','strong_300')]
    cfg['role']='PI native refinement: same C2 turning points; both preparation and coupling refined'
    out=args.out.resolve()
    if out==ROOT/'results' or not out.is_relative_to(ROOT/'results') or out.exists():
        raise ValueError('Use a fresh results subdirectory')
    out.mkdir(parents=True)
    paths=source_paths(source)+[Path(__file__).resolve(),REPO/'study_v2/configs/experiment.json']
    hashes={str(p):sha(p) for p in paths}
    write_json(out/'generation_lock.json',dict(config=cfg,sources_sha256=hashes,smoke=False,
        divisor=args.divisor,role=cfg['role'],command=sys.argv),exclusive=True)
    start=time.perf_counter();app=None;batches=episodes=0
    process=dict(status='starting',pid=os.getpid(),started_unix=time.time(),command=sys.argv)
    write_json(out/'process.json',process,exclusive=True)
    try:
        app=AppLauncher(args).app
        from decision_study.isaac_rig import IsaacRig
        rig=IsaacRig(cfg['episodes_per_batch'],'cpu',dt=cfg['dt'])
        write_json(out/'process.json',{**process,'status':'running'})
        for seed in cfg['preparation_seeds']:
            factors=factorial_design(seed)
            for normal in cfg['normal_worlds']:
                for case in cfg['friction_parameter_sets']:
                    for probe in cfg['probes']:
                        run_batch(rig,normal,case,probe,'ideal',cfg,factors,
                            out/f"p{seed}/{normal}_{case}/{probe['name']}/ideal.npz",deadline=start+1800.)
                        batches+=1;episodes+=len(factors)
        if hashes!={p:sha(p) for p in hashes}:raise RuntimeError('Source changed during native refinement')
        write_json(out/'process.json',{**process,'status':'complete','batches':batches,
            'episodes':episodes,'seconds':time.perf_counter()-start})
        print('PI REFINEMENT COMPLETE',out,flush=True)
    except BaseException as error:
        write_json(out/'process.json',{**process,'status':'failed','completed_batches':batches,
            'error':repr(error),'seconds':time.perf_counter()-start})
        raise
    finally:
        if app is not None:app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=='__main__':main()

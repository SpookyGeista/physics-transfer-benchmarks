"""Native h/2 and h/4 check of the crossed development rig, ideal sensors only.

The material-fixture prehistory remains the identical, already declared 2.5-ms
protocol. Only subsequent native motion/material integration is refined. This
does not constitute a convergence study of the external preparation fixture.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("NUMBA_CACHE_DIR","/tmp/paper-study-numba")
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse
import json
import time
from generate_native_v2 import (ROOT, sha, source_paths, write_json, run_batch, factorial_design,
                                validate_config)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--divisor",type=int,choices=[2,4],required=True)
    parser.add_argument("--out",type=Path,required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    config_path=ROOT/"configs/development_v2.json"
    cfg=json.loads(config_path.read_text())
    validate_config(cfg)
    if cfg["split"]!="dev" or args.device!="cpu":
        raise ValueError("This refinement is CPU development only")
    cfg["dt"]/=args.divisor
    cfg["observation_steps"]*=args.divisor
    cfg["sensor_conditions"]={"ideal":cfg["sensor_conditions"]["ideal"]}
    cfg["probes"]=[p for p in cfg["probes"] if p["name"] in ("none","hold_300","gentle_150","strong_300")]
    cfg["role"]="Ideal-sensor native timestep refinement with fixed external material preparation"
    out=args.out.resolve()
    if (not out.is_relative_to(ROOT/"results") or out==ROOT/"results"
            or out.exists() and any(out.iterdir())):
        raise ValueError("Use a fresh result subdirectory")
    out.mkdir(parents=True,exist_ok=True)
    sources=source_paths(config_path)+[Path(__file__).resolve()]
    hashes={str(p):sha(p) for p in sources}
    write_json(out/"generation_lock.json",dict(sources_sha256=hashes,config=cfg,
        command=sys.argv,role=cfg["role"],divisor=args.divisor),exclusive=True)
    start=time.perf_counter(); app=None
    process=dict(pid=os.getpid(),status="starting",started_unix=time.time(),command=sys.argv)
    write_json(out/"process.json",process,exclusive=True)
    batches=episodes=0
    try:
        app=AppLauncher(args).app
        from decision_study.isaac_rig import IsaacRig
        rig=IsaacRig(cfg["episodes_per_batch"],"cpu",dt=cfg["dt"])
        for seed in cfg["preparation_seeds"]:
            factors=factorial_design(seed,cfg["split"],cfg["cam_amplitudes_m"],cfg["excitation_signs"])
            for nw,fw in cfg["compositions"]:
                for probe in cfg["probes"]:
                    run_batch(rig,nw,fw,probe,"ideal",cfg,factors,
                        out/f"p{seed}/{nw}_{fw}/{probe['name']}/ideal.npz",deadline=start+1800.)
                    batches+=1; episodes+=len(factors)
        if hashes!={p:sha(p) for p in hashes}:
            raise RuntimeError("Source changed during refinement")
        write_json(out/"process.json",dict(**{**process,"status":"complete"},
            batches=batches,episodes=episodes,seconds=time.perf_counter()-start))
        print("REFINEMENT COMPLETE",out,flush=True)
    except BaseException as exc:
        write_json(out/"process.json",dict(**{**process,"status":"failed"},error=repr(exc),
            completed_batches=batches,seconds=time.perf_counter()-start))
        raise
    finally:
        if app is not None: app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=="__main__":main()

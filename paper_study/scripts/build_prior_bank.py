"""Population priors from public preparation protocol and frozen local models.

No native trial or reference force/state is read. The protocol's prescribed
velocity/load-distribution is public; the individual realized prehistory is not.
Every hypothesis is produced by fitted dynamics, then projected to a fixed grid.
An explicit broad component protects against unseen histories. This is a model
of the declared preparation population, not an empirically validated real-world
history distribution.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/paper-study-numba")
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from models import load_model, model_paths
from preparation import preparation_plan


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_mass(samples, grid):
    """Linear mass splitting in the declared grid coordinate, never KDE tuning."""
    if np.any(samples < grid[0]) or np.any(samples > grid[-1]):
        raise ValueError("Fitted protocol states exceed declared grid support")
    hi = np.minimum(np.searchsorted(grid, samples, side="right"), len(grid)-1)
    lo = np.maximum(hi-1, 0)
    fraction = (samples-grid[lo])/(grid[hi]-grid[lo])
    result = np.zeros(len(grid))
    np.add.at(result, lo, 1-fraction)
    np.add.at(result, hi, fraction)
    return result/len(samples)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/prior_development.json")
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    out=args.out.resolve()
    if not out.is_relative_to(ROOT/"results") or (out.exists() and any(out.iterdir())):
        raise ValueError("Need a fresh output directory under paper_study/results")
    sources=[Path(__file__),args.config,ROOT/"src/models.py",ROOT/"src/preparation.py",
             ROOT.parent/"research_next/src/state_id.py"]
    for nw,fw in cfg["compositions"]:
        for seed in cfg["model_seeds"]:
            for noise in cfg["calibration_noise"]:
                sources.extend(model_paths(nw,fw,seed,noise))
    hashes={str(p.resolve()):sha(p) for p in sources}
    out.mkdir(parents=True)
    (out/"prior_lock.json").write_text(json.dumps(dict(config=cfg,sources_sha256=hashes,
        role="Prior generation locked before execution; no task observations or truth states"),indent=2)+"\n")
    start=time.perf_counter()
    plan=preparation_plan(cfg["protocol_samples"],cfg["protocol_seed"],"dev")
    for nw,fw in cfg["compositions"]:
        for seed in cfg["model_seeds"]:
            for noise in cfg["calibration_noise"]:
                model=load_model(nw,fw,seed,noise)
                n=cfg["protocol_samples"]
                delta=np.full(n,.02); zeros=np.zeros(n)
                normal_state=model.normal_model.initial(delta,zeros,zeros)
                initial_n=model.normal_model.step(delta,zeros,zeros,normal_state)[0]
                friction=model.friction_model
                state=friction.initial(delta,zeros,initial_n)
                for t in range(len(plan.time)-1):
                    _,_,state=friction.step(delta,zeros,plan.velocity[:,t],state,
                        normal_override=initial_n*plan.normal_scale[:,t],dt=cfg["dt"])
                key="mix" if fw=="maxwell" else "age"
                spec=cfg["prior"][fw]
                space=np.linspace if spec["spacing"]=="linear" else np.geomspace
                grid=space(*spec["bounds"],cfg["grid_size"])
                # For log spacing use the corresponding log coordinate to split mass.
                transform=(lambda x:x) if spec["spacing"]=="linear" else np.log
                mass=project_mass(transform(state[key]),transform(grid))
                broad=cfg["broad_prior_fraction"]
                weights=(1-broad)*mass+broad/len(grid)
                dest=out/f"{nw}_{fw}/r{seed}/{noise}.npz"
                dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open("xb") as f:
                    np.savez_compressed(f,grid=grid,weights=weights,sampled_initial_state=state[key])
                print(f"PRIOR {nw}/{fw} r{seed}/{noise} range={state[key].min():.5g}..{state[key].max():.5g}",flush=True)
    assert hashes=={p:sha(p) for p in hashes}
    (out/"process.json").write_text(json.dumps(dict(status="complete",seconds=time.perf_counter()-start,
        bank_files=len(list(out.rglob('*.npz')))),indent=2)+"\n")


if __name__=="__main__":
    main()

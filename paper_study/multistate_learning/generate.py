"""Truth-side local data generation, executed separately from the fitter."""
import json
from pathlib import Path
import sys
import time

import numpy as np

from lock import ROOT, REPO, digest, verify
from profiles import profiles

sys.path.insert(0, str(REPO/'paper_study/alternatives'))
from multistate_friction import MultiStateFriction


def main():
    start = time.monotonic()
    source_lock = verify()
    cfg = json.loads((ROOT/'config.json').read_text())
    out = ROOT/'results/data'
    if out.exists():
        raise RuntimeError('refusing to overwrite any existing calibration directory')
    manifest = dict(design_lock_sha256=source_lock, files={})
    for case_index, case in enumerate(cfg['cases']):
        for seed in cfg['replicate_seeds']:
            for split_index, split in enumerate(('train', 'val')):
                commands, rows = profiles(cfg, seed, split)
                v, normal = commands['v'], commands['N']
                truth = MultiStateFriction(case, len(v))
                force = np.empty_like(v)
                for t in range(v.shape[1]):
                    force[:, t] = truth.force(v[:, t], normal[:, t], cfg['dt'])
                rng = np.random.default_rng(np.random.SeedSequence(
                    [cfg['noise_seed_base'], seed, case_index, split_index]))
                standard_noise = rng.normal(size=force.shape)
                for condition, sigma in cfg['conditions'].items():
                    folder = out/case/f'r{seed}'/condition
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder/f'{split}.npz'
                    # Fitter's complete allowed schema: v,N,Ft,dt only.
                    np.savez_compressed(path, v=v, N=normal, Ft=force+sigma*standard_noise,
                                        dt=np.array(cfg['dt']))
                    meta = dict(case=case, seed=seed, condition=condition, split=split,
                                sigma_force_N=sigma, initial_state='known relaxed',
                                sensor_channels=['v','N','Ft'], rows=rows,
                                design_lock_sha256=source_lock, npz_sha256=digest(path))
                    json_path = path.with_suffix('.json')
                    json_path.write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')
                    for p in (path, json_path):
                        manifest['files'][str(p.relative_to(ROOT))] = digest(p)
            print(f'generated {case} r{seed}', flush=True)
    manifest['wall_seconds'] = time.monotonic()-start
    verify()
    (ROOT/'results/dataset_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'data complete in {manifest["wall_seconds"]:.2f}s', flush=True)


if __name__ == '__main__':
    main()

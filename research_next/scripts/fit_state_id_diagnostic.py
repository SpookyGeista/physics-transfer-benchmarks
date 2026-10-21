"""Bounded isolated train/validation identification, never a main-test reader."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from state_id import StateID, fit_state_id, fit_blind_state_id


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(root, world, seed, split, noise):
    if split not in ('train', 'val'):
        raise ValueError('diagnostic fitting may read only isolated train/val')
    path = Path(root)/world/f'r{seed}'/(split+'.npz')
    z = dict(np.load(path)); metadata = json.loads(path.with_suffix('.json').read_text())
    rows = metadata['rows']; assert len(rows) == len(z['delta'])
    assert all(row['mode'] in ('A', 'B') for row in rows)
    ni = np.array([row['mode'] == 'B' for row in rows]); fi = ~ni
    delta = np.asarray(z['delta'], dtype=float).copy()
    rate = np.asarray(z.get('compression_rate', z['rate']), dtype=float).copy()
    velocity = np.asarray(z['state'][..., 1], dtype=float).copy()
    normal = np.asarray(z['normal'], dtype=float).copy()
    friction = np.asarray(z['friction'], dtype=float).copy()
    if noise == 'noisy':
        rng = np.random.default_rng(7200000+seed*100+{'train': 1, 'val': 2}[split])
        # Same measurement order and declared magnitudes as the old isolated run.
        delta += rng.normal(0, .00005, delta.shape)
        rate += rng.normal(0, .001, rate.shape)
        velocity += rng.normal(0, .002, velocity.shape)
        normal += rng.normal(0, .15, normal.shape)
        friction += rng.normal(0, .10, friction.shape)
    data = dict(normal=dict(delta=delta[ni], rate=rate[ni], target=normal[ni]),
                friction=dict(delta=delta[fi], rate=rate[fi], velocity=velocity[fi],
                              normal=normal[fi], target=friction[fi]))
    return data, dict(npz=str(path.resolve()), npz_sha256=sha(path),
                     json_sha256=sha(path.with_suffix('.json')), ids=[row['id'] for row in rows],
                     dt=float(metadata['dt']))


def evaluate_local(model, data, dt):
    bn, af = data['normal'], data['friction']
    nstate = model.initial(bn['delta'][:, 0], np.zeros(len(bn['delta'])), bn['target'][:, 0])
    fstate = model.initial(af['delta'][:, 0], af['velocity'][:, 0], af['normal'][:, 0])
    pn, pf = [], []
    for t in range(bn['target'].shape[1]):
        n, _, nstate = model.step(bn['delta'][:, t], bn['rate'][:, t], np.zeros(len(bn['delta'])), nstate, dt=dt)
        pn.append(n)
    for t in range(af['target'].shape[1]):
        _, ft, fstate = model.step(af['delta'][:, t], af['rate'][:, t], af['velocity'][:, t], fstate,
                                  normal_override=af['normal'][:, t], dt=dt)
        pf.append(ft)
    pn, pf = np.stack(pn, 1), np.stack(pf, 1)
    return dict(normal=pn, friction=pf), dict(normal_rmse=float(np.sqrt(np.mean((pn-bn['target'])**2))),
                                           friction_rmse=float(np.sqrt(np.mean((pf-af['target'])**2))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT.parent/'study_v2/results/data')
    parser.add_argument('--out', type=Path, default=ROOT/'results/state_id_diagnostic')
    parser.add_argument('--worlds', nargs='+', choices=['maxwell', 'boucwen'], default=['maxwell', 'boucwen'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--noise', nargs='+', choices=['clean', 'noisy'], default=['clean'])
    parser.add_argument('--max-seconds', type=float, default=120.)
    parser.add_argument('--restarts', type=int, default=3)
    parser.add_argument('--blind-selection', action='store_true',
                        help='fit both candidate families on every dataset; isolated val selects each head')
    args = parser.parse_args()
    code = {str(path): sha(path) for path in [Path(__file__), ROOT/'src/state_id.py']}
    for world in args.worlds:
        for seed in args.seeds:
            for noise in args.noise:
                train, ts = load_data(args.data, world, seed, 'train', noise)
                val, vs = load_data(args.data, world, seed, 'val', noise)
                assert ts['dt'] == vs['dt'] and set(ts['ids']).isdisjoint(vs['ids'])
                spec = dict(world=world, seed=seed, noise=noise, train=ts, validation=vs, code=code,
                            max_seconds=args.max_seconds, restarts=args.restarts, blind_selection=args.blind_selection)
                fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
                dest = args.out/world/f'r{seed}'/noise; dest.mkdir(parents=True, exist_ok=True)
                if args.blind_selection:
                    dest = dest/'blind_selection'; dest.mkdir(exist_ok=True)
                if (dest/'metadata.json').exists():
                    old = json.loads((dest/'metadata.json').read_text())
                    assert old['fingerprint'] == fingerprint and old['model_sha256'] == sha(dest/'model.json'), 'changed diagnostic inputs; use fresh --out'
                    print('SKIP verified', dest, flush=True); continue
                assert not (dest/'model.json').exists(), 'partial output; use fresh --out and retain old artifacts'
                print('FIT', world, seed, noise, flush=True)
                if args.blind_selection:
                    # `world` labels only the input/output path; selection receives no such argument.
                    model = fit_blind_state_id(train, val, dt=ts['dt'], max_seconds=args.max_seconds,
                                              seed=seed, restarts=args.restarts)
                else:
                    model = fit_state_id(world, train, val, dt=ts['dt'], max_seconds=args.max_seconds,
                                         seed=seed, restarts=args.restarts)
                model.save(dest/'model.json')
                trpred, trmetrics = evaluate_local(model, train, ts['dt'])
                vapred, vametrics = evaluate_local(model, val, ts['dt'])
                np.savez_compressed(dest/'validation_predictions.npz', **vapred,
                                    measured_normal=val['normal']['target'], measured_friction=val['friction']['target'])
                metadata = dict(**spec, fingerprint=fingerprint, model_sha256=sha(dest/'model.json'),
                                train_metrics=trmetrics, validation_metrics=vametrics, fit=model.metadata,
                                scope='exploratory isolated diagnostic; stronger known-family prior; no main-test access')
                (dest/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
                print('DONE', world, seed, noise, 'train', trmetrics, 'validation', vametrics,
                      'seconds', model.metadata['seconds'], flush=True)


if __name__ == '__main__':
    main()

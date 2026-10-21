"""Independent local input design. No reference/learner imports or parameters."""
import numpy as np


def profiles(config, seed, split):
    if split not in ('train', 'val'):
        raise ValueError(split)
    per_load = config[f'{split}_per_load']
    dt, segment = config['dt'], config['segment_s']
    ns = int(round(segment/dt))
    s = np.arange(ns+1)/ns
    blend = s**3*(10-15*s+6*s*s)
    arrays = {name: [] for name in ('v', 'N', 'x')}
    rows = []
    for load_index, load in enumerate(config['normal_loads_N']):
        for profile_index in range(per_load):
            rng = np.random.default_rng(np.random.SeedSequence(
                [config['profile_seed_bases'][split], seed, load_index, profile_index]))
            if split == 'train':
                peak = config['train_peak_displacements_m'][profile_index] * rng.uniform(.85, 1.15)
            else:
                peak = float(rng.uniform(.004, .036))
            sign = rng.choice([-1., 1.])
            # Distinct random reversal ratios; final stationary segment exposes
            # residual force. Every sequence starts from known relaxed state.
            a, b, c = rng.uniform(.35, .85), rng.uniform(.15, .65), rng.uniform(.05, .35)
            points = sign*peak*np.array([0., 1., -a, b, -c, 0., 0.])
            x = [0.]
            for first, last in zip(points[:-1], points[1:]):
                x.extend((first+(last-first)*blend)[1:])
            x = np.asarray(x)
            velocity = np.diff(x)/dt
            arrays['v'].append(velocity)
            arrays['N'].append(np.full_like(velocity, load))
            arrays['x'].append(x)
            rows.append(dict(split=split, seed=seed, load_N=load,
                             profile_index=profile_index, waypoints_m=points.tolist()))
    return {key: np.array(value) for key, value in arrays.items()}, rows

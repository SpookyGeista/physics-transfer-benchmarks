"""Public design and commands. No reference, preparation identity or target data."""
from pathlib import Path
import hashlib
import json
import numpy as np
from paper_study.costed_benchmark.geometry import cam

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
BRANCHES=('release','hold100','continue100')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value


def write_json(path,value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n')


def load_config(path=None):
    cfg=json.loads(Path(path or HERE/'config.json').read_text())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    required_fixed={'preparation_seeds':[34001,34002], 'profiles_per_seed':6,
        'geometry_ids':[0,1,2], 'excitation_signs':[-1,1], 'dt':.0025,'mass_kg':2.,
        'initial_compression_m':.02,'observation_steps':10,'common_seconds':.05,
        'continuation_seconds':.1,'probe_period_seconds':.15,'probe_excitation_N':3.,
        'future_seconds':.25,'future_excitation_N':5.,'future_compression_amplitude_m':.002,
        'cam_amplitude_m':.003,'kp':120.,'kd':20.,'force_limit_N':12.,
        'sensor_sigma':[5e-5,.002,.15], 'numerical_floor':[1e-6,1e-5,1e-4],
        'particles':201,'fantasies':64,'model_seed':1,'calibration_noise':'noisy'}
    for key,value in required_fixed.items():
        if cfg.get(key)!=value:raise ValueError(f'Unsupported declared setting: {key}')
    if cfg['compositions']!=[[a,b] for a in ('maxwell','boucwen') for b in ('maxwell','boucwen')]:
        raise ValueError('All four declared compositions required')
    for key in ('native_wall_seconds','evaluation_wall_seconds'):
        if not np.isfinite(cfg[key]) or cfg[key]<=0:raise ValueError(key)
    for key in ('sensor_seed','fantasy_seed','policy_seed'):
        if isinstance(cfg[key],bool) or not isinstance(cfg[key],int) or cfg[key]<0:raise ValueError(key)
    expected=[('B0',0.,0.),('B1',1.,.1),('B2',1.,.5),('B3',1.,2.)]
    actual=[(b['name'],b['max_scaled_time'],b['max_scaled_work']) for b in cfg['budgets']]
    if actual!=expected or any(b['time_scale_s']!=.1 or b['work_scale_J']!=.01 for b in cfg['budgets']):
        raise ValueError('Use all four declared dimensionless expected-cost budgets')


def steps(cfg):
    common=round(cfg['common_seconds']/cfg['dt'])
    later=common+round(cfg['continuation_seconds']/cfg['dt'])
    future=round(cfg['future_seconds']/cfg['dt'])
    return common,later,future


def feedback(observed,amplitude,geometry,cfg):
    observed=np.asarray(observed,dtype=float)
    if observed.ndim!=2 or observed.shape[1]!=3 or not np.isfinite(observed).all():
        raise ValueError('finite episodes x (x,v,N) required')
    bias=cam(observed[:,0],amplitude,geometry)[1]*observed[:,2]
    return bias-cfg['kp']*observed[:,0]-cfg['kd']*observed[:,1],bias


def force_before_decision(step,held_feedback,signs,cfg):
    return np.clip(held_feedback+np.asarray(signs)*cfg['probe_excitation_N']*
        np.sin(2*np.pi*step*cfg['dt']/cfg['probe_period_seconds']),
        -cfg['force_limit_N'],cfg['force_limit_N'])


def branch_commands(branch,held_feedback,bias,signs,cfg):
    """All commands AFTER t=50ms; no subsequent measurement argument exists."""
    if branch not in BRANCHES:raise ValueError(branch)
    common,later,future=steps(cfg)
    release=common if branch=='release' else later
    n=len(np.atleast_1d(bias));total=release+future
    u=np.empty((n,total-common));d=np.empty_like(u);rate=np.empty_like(u)
    for j,t in enumerate(range(common,total)):
        if t<release:
            ff=0. if branch=='hold100' else np.asarray(signs)*cfg['probe_excitation_N']*np.sin(
                2*np.pi*t*cfg['dt']/cfg['probe_period_seconds'])
            u[:,j]=np.clip(held_feedback+ff,-cfg['force_limit_N'],cfg['force_limit_N'])
            d[:,j]=cfg['initial_compression_m'];rate[:,j]=0.
        else:
            phase=2*np.pi*(t-release)*cfg['dt']/cfg['future_seconds']
            u[:,j]=np.clip(bias+np.asarray(signs)*cfg['future_excitation_N']*np.sin(phase),
                          -cfg['force_limit_N'],cfg['force_limit_N'])
            d[:,j]=cfg['initial_compression_m']+cfg['future_compression_amplitude_m']*np.sin(phase)
            rate[:,j]=cfg['future_compression_amplitude_m']*2*np.pi/cfg['future_seconds']*np.cos(phase)
    return dict(u=u,d=d,d_dot=rate,release_step=release,common_step=common)


def fresh_output(path):
    path=Path(path).resolve()
    if not path.is_relative_to(HERE) or path==HERE or path.exists():
        raise ValueError('Use a fresh output subdirectory inside decision_study')
    path.mkdir(parents=True)
    return path


def verify_hashes(mapping):
    for path,digest in mapping.items():
        if sha(path)!=digest:raise ValueError(f'Source/input hash mismatch: {path}')

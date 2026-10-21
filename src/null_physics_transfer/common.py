import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / 'configs/experiment.json').read_text())
DT = CFG['dt']
TAUS = np.array(CFG['history_taus'])
MASS = CFG['mass']
VREG = CFG['velocity_regularization']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))

def history_update(h, delta, velocity, dt=DT):
    values = np.stack([np.maximum(delta,0), velocity], -1)
    return h + (1-np.exp(-dt/TAUS))[None,:,None] * (values[:,None,:]-h)

def features(delta, rate, velocity, h, normal):
    rate=np.where(delta>0,rate,0); delta=np.maximum(delta,0)
    # All scales are declared unit scales, never computed from test data.
    return np.column_stack([delta/.02, rate/.1, velocity/.5,
        (delta[:,None]-h[:,:,0])/.02, (velocity[:,None]-h[:,:,1])/.5, normal/20]).astype(np.float32)

def design(split, count=None):
    """Whole trajectories with disjoint random seeds and arrangement families."""
    base = {'train':1000, 'val':9000, 'privileged':31000, 'test':CFG['test_seed'], 'control':81000}[split]
    rows=[]
    if split in ('train','val'):
        n = count or (CFG['calibration_per_effect'] if split=='train' else CFG['validation_per_effect'])
        for mode in ('A','B'):
            for i in range(n):
                rows.append(dict(id=f'{split}-{mode}-{i}', mode=mode, slope=0., seed=base+100*(mode=='B')+i, condition=mode))
    elif split=='privileged':
        for i in range(CFG['privileged_episodes']):
            rows.append(dict(id=f'privileged-AB-{i}',mode='AB',slope=[-.06,.06][i%2],seed=base+i,condition='privileged'))
    else:
        for c,s in enumerate(CFG['slopes']):
            for i in range(count or CFG['test_per_condition']):
                rows.append(dict(id=f'{split}-{c}-{i}',mode='AB',slope=s,seed=base+c*100+i,condition=f'cam_{s:+.2f}'))
    return rows

def commands(rows, steps=None, dt=DT):
    n=len(rows); t=np.arange(steps or CFG['steps'])*dt
    u=np.zeros((n,len(t))); d=u.copy(); rate=u.copy(); prescribed_v=u.copy(); loads=u.copy()
    for i,row in enumerate(rows):
        rng=np.random.default_rng(row['seed'])
        phase=rng.uniform(0,2*np.pi,3); freq=rng.uniform(.35,1.7,3)
        # Absolute cam shape and actuator waveform are known, not learned.
        u[i]=rng.uniform(5.,9.)*np.sin(2*np.pi*freq[0]*t+phase[0])+2*np.sin(2*np.pi*freq[1]*t+phase[1])
        d[i]=rng.uniform(.009,.018)+.005*np.sin(2*np.pi*freq[2]*t+phase[2])
        rate[i]=.005*2*np.pi*freq[2]*np.cos(2*np.pi*freq[2]*t+phase[2])
        prescribed_v[i]=rng.uniform(.15,.6)*np.sin(2*np.pi*freq[0]*t+phase[0])+.07*np.sin(2*np.pi*freq[1]*t+phase[1])
        loads[i]=rng.uniform(5.,24.)
    return dict(u=u,d=d,rate=rate,prescribed_v=prescribed_v,loads=loads)

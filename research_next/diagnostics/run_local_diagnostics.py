"""Bounded exploratory local-only representation and optimization diagnostics.

Uses the old v2 isolated train/val libraries (now development data), r0 only.
No material reference imports, new primary test, controller, or robot interfaces.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/physics-local-diagnostic-matplotlib')
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
V2 = REPO/'study_v2'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2/'src'))
from decision_study.models import load_model

DT = .0025
TAUS_OLD = [.02, .08, .3]
TAUS_LONG = [.005, .01, .02, .04, .08, .16, .32, .64, 1.28]
VARIANTS = [dict(name='base3_longopt', feature='base3', width=24),
            dict(name='bank9', feature='bank9', width=24),
            dict(name='bank9_path', feature='bank9_path', width=24),
            dict(name='bank9_path_wide', feature='bank9_path', width=64)]
UPDATES = 12000
LAMBDAS = [1e-8, 1e-6, 1e-4, .01, 1.]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))


def load_data(world, split):
    assert split in ['train', 'val']
    path = V2/f'results/data/{world}/r0/{split}.npz'
    with np.load(path) as arrays:
        data = {k: arrays[k] for k in ['delta', 'rate', 'state', 'normal', 'friction', 'features']}
    rows = json.loads(path.with_suffix('.json').read_text())['rows']
    assert all(row['mode'] in ['A', 'B'] for row in rows)
    return data, rows


def build_features(data, kind):
    """Every feature at t uses only current observables and earlier observations.

    Normal features never use any current/past normal or friction measurement.
    Friction current/past N is allowed measured local interface calibration.
    Running extrema include the observed current d; integrals exclude current t.
    """
    if kind == 'base3':
        f = data['features']
        return f[:, :, [0, 1, 3, 4, 5]], f[:, :, [2, 6, 7, 8, 9, 10, 11, 12]]
    d, r, v, n = data['delta'], data['rate'], data['state'][:, :, 1], data['normal']
    count, steps = d.shape; taus = np.array(TAUS_LONG)
    initial_d = d[:, 0].copy(); dmin = initial_d.copy(); dmax = initial_d.copy()
    values0 = np.column_stack([d[:, 0], v[:, 0], n[:, 0], np.abs(v[:, 0])])
    h = np.repeat(values0[:, None, :], len(taus), axis=1)
    travel_d = np.zeros(count); slip = np.zeros(count)
    normals, frictions = [], []
    decay = (1-np.exp(-DT/taus))[None, :, None]
    for t in range(steps):
        nf = np.column_stack([d[:, t]/.02, r[:, t]/.1, (d[:, t, None]-h[:, :, 0])/.02])
        ff = np.column_stack([v[:, t]/.5, (v[:, t, None]-h[:, :, 1])/.5,
                              n[:, t]/20, (n[:, t, None]-h[:, :, 2])/20])
        dmin = np.minimum(dmin, d[:, t]); dmax = np.maximum(dmax, d[:, t])
        if kind == 'bank9_path':
            nf = np.column_stack([nf, initial_d/.02, dmin/.02, dmax/.02, travel_d/.02])
            ff = np.column_stack([ff, np.full(count, t*DT/2), slip/.1, np.abs(v[:, t])/.5, h[:, :, 3]/.5])
        normals.append(nf); frictions.append(ff)
        h += decay*(np.column_stack([d[:, t], v[:, t], n[:, t], np.abs(v[:, t])])[:, None, :]-h)
        travel_d += DT*np.abs(r[:, t]); slip += DT*np.abs(v[:, t])
    return np.stack(normals, 1).astype(np.float32), np.stack(frictions, 1).astype(np.float32)


def local_data(data, rows, kind):
    nf, ff = build_features(data, kind)
    ni = [i for i, row in enumerate(rows) if row['mode'] == 'B']
    fi = [i for i, row in enumerate(rows) if row['mode'] == 'A']
    return dict(normal=nf[ni].reshape(-1, nf.shape[-1]), friction=ff[fi].reshape(-1, ff.shape[-1]),
                target_n=data['normal'][ni].ravel(), target_f=data['friction'][fi].ravel(),
                # Explicit physical factor avoids target division at zero speed.
                friction_factor=(-data['normal'][fi]*np.tanh(data['state'][fi, :, 1]/.04)).ravel(),
                normal_ids=[rows[i]['id'] for i in ni], friction_ids=[rows[i]['id'] for i in fi], steps=data['delta'].shape[1])


def network(inputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))


class LocalModel(nn.Module):
    def __init__(self, nd, fd, width):
        super().__init__(); self.normal_net = network(nd, width); self.friction_net = network(fd, width)
    def normal(self, x): return torch.relu(10*x[:, 0]+20*self.normal_net(x)[:, 0])*(x[:, 0] > 0)
    def friction(self, x, factor): return factor*nn.functional.softplus(self.friction_net(x)[:, 0]-.87)


def evaluate(model, dataset):
    with torch.no_grad():
        n = model.normal(torch.as_tensor(dataset['normal'])).numpy()
        f = model.friction(torch.as_tensor(dataset['friction']), torch.as_tensor(dataset['friction_factor'], dtype=torch.float32)).numpy()
    return n, f


def errors(n, f, dataset):
    en, ef = n-dataset['target_n'], f-dataset['target_f']
    return dict(normal_rmse=float(np.sqrt(np.mean(en**2))), friction_rmse=float(np.sqrt(np.mean(ef**2))),
                scaled_loss=float(np.mean((en/20)**2)+np.mean((ef/10)**2)),
                normal_profile_rmse=np.sqrt(np.mean(en.reshape(-1, dataset['steps'])**2, axis=1)).tolist(),
                friction_profile_rmse=np.sqrt(np.mean(ef.reshape(-1, dataset['steps'])**2, axis=1)).tolist())


def fit_mlp(world, variant, train, val):
    dest = OUT/'runs'/world/variant['name']; dest.mkdir(parents=True, exist_ok=True)
    if (dest/'metrics.json').exists(): return json.loads((dest/'metrics.json').read_text())
    torch.manual_seed(0)
    model = LocalModel(train['normal'].shape[1], train['friction'].shape[1], variant['width'])
    optimizer = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=1e-6)
    tn, tf = torch.as_tensor(train['normal']), torch.as_tensor(train['friction'])
    yn, yf = torch.as_tensor(train['target_n'], dtype=torch.float32), torch.as_tensor(train['target_f'], dtype=torch.float32)
    factor = torch.as_tensor(train['friction_factor'], dtype=torch.float32)
    start = time.perf_counter(); curves = []; best = float('inf'); chosen = None; checkpoint_update = None
    for update in range(UPDATES):
        # Fixed staged schedule, declared before any diagnostic outcome.
        if update == 6000:
            for group in optimizer.param_groups: group['lr'] = .0005
        ni, fi = torch.randint(len(tn), (512,)), torch.randint(len(tf), (512,))
        loss = ((model.normal(tn[ni])-yn[ni])/20).square().mean()+((model.friction(tf[fi], factor[fi])-yf[fi])/10).square().mean()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
        if update % 200 == 0 or update in [1199, UPDATES-1]:
            vn, vf = evaluate(model, val); ve = errors(vn, vf, val)
            curves.append(dict(update=update+1, training_batch_loss=float(loss.item()), validation=ve, seconds=time.perf_counter()-start))
            if ve['scaled_loss'] < best:
                best = ve['scaled_loss']; chosen = copy.deepcopy(model.state_dict()); checkpoint_update = update+1
    model.load_state_dict(chosen); model.eval()
    pn, pf = evaluate(model, train); vn, vf = evaluate(model, val)
    metrics = dict(world=world, kind='mlp', variant=variant['name'], feature=variant['feature'], width=variant['width'],
                   normal_inputs=train['normal'].shape[1], friction_inputs=train['friction'].shape[1],
                   parameters=sum(p.numel() for p in model.parameters()), updates=UPDATES, selected_update=checkpoint_update,
                   seconds=time.perf_counter()-start, train=errors(pn, pf, train), val=errors(vn, vf, val))
    torch.save(dict(state=chosen, variant=variant, dimensions=[train['normal'].shape[1], train['friction'].shape[1]]), dest/'model.pt')
    np.savez_compressed(dest/'predictions.npz', train_normal=pn, train_friction=pf, val_normal=vn, val_friction=vf,
                        val_true_normal=val['target_n'], val_true_friction=val['target_f'])
    dump(dest/'learning.json', curves); dump(dest/'metrics.json', metrics)
    print('MLP_DONE', world, variant['name'], metrics, flush=True)
    return metrics


def basis(x, degree, absolute=False):
    x = x.astype(np.float64); cols = [np.ones(len(x))]
    for order in range(1, degree+1):
        cols.extend(np.prod(x[:, indices], axis=1) for indices in itertools.combinations_with_replacement(range(x.shape[1]), order))
    if absolute: cols.extend(np.abs(x[:, i]) for i in range(x.shape[1]))
    return np.column_stack(cols)


def fit_ridge(world, feature, train, val):
    dest = OUT/'runs'/world/('ridge_'+feature); dest.mkdir(parents=True, exist_ok=True)
    if (dest/'metrics.json').exists(): return json.loads((dest/'metrics.json').read_text())
    started = time.perf_counter(); fitted = {}; predictions = {}
    for head, degree in [('normal', 3), ('friction', 2)]:
        b = basis(train[head], degree, head == 'friction'); vb = basis(val[head], degree, head == 'friction')
        mean, scale = b.mean(0), b.std(0); mean[0], scale[0] = 0, 1; scale = np.where(scale < 1e-8, 1, scale)
        z, vz = (b-mean)/scale, (vb-mean)/scale
        factors = [np.ones(len(z)), np.ones(len(vz))] if head == 'normal' else [train['friction_factor']/10, val['friction_factor']/10]
        target = train['target_n']/20 if head == 'normal' else train['target_f']/10
        wz = z*factors[0][:, None]; gram, rhs = wz.T@wz/len(z), wz.T@target/len(z)
        reg = np.eye(z.shape[1]); reg[0, 0] = 0
        candidates = []
        for strength in LAMBDAS:
            coef = np.linalg.solve(gram+strength*reg+1e-12*np.eye(len(reg)), rhs)
            if head == 'normal':
                vp = 20*np.maximum(vz@coef, 0); yt = val['target_n']
            else:
                vp = val['friction_factor']*np.maximum(vz@coef, 0); yt = val['target_f']
            candidates.append((float(np.mean((vp-yt)**2)), strength, coef, vp))
        loss, strength, coef, vp = min(candidates, key=lambda x: x[0])
        tp = 20*np.maximum(z@coef, 0) if head == 'normal' else train['friction_factor']*np.maximum(z@coef, 0)
        predictions['train_'+head], predictions['val_'+head] = tp, vp
        fitted[head] = dict(mean=mean.tolist(), scale=scale.tolist(), coef=coef.tolist(), selected_lambda=strength,
                            candidates=[dict(lambda_=a[1], val_rmse=float(np.sqrt(a[0]))) for a in candidates])
    metrics = dict(world=world, kind='ridge', variant='ridge_'+feature, feature=feature,
                   parameters=sum(len(fitted[h]['coef']) for h in fitted), updates=0,
                   normal_inputs=train['normal'].shape[1], friction_inputs=train['friction'].shape[1],
                   seconds=time.perf_counter()-started,
                   train=errors(predictions['train_normal'], predictions['train_friction'], train),
                   val=errors(predictions['val_normal'], predictions['val_friction'], val),
                   lambdas={head: fitted[head]['selected_lambda'] for head in fitted})
    dump(dest/'model.json', fitted); dump(dest/'metrics.json', metrics)
    np.savez_compressed(dest/'predictions.npz', **predictions, val_true_normal=val['target_n'], val_true_friction=val['target_f'])
    print('RIDGE_DONE', world, feature, metrics, flush=True)
    return metrics


def test_causality():
    shape = (2, 16)
    data = dict(delta=np.full(shape, .02), rate=np.ones(shape)*.01, state=np.zeros((*shape, 2)), normal=np.full(shape, 10.))
    original_n, original_f = build_features(data, 'bank9_path')
    altered = {k: v.copy() for k, v in data.items()}; altered['normal'] += 100
    assert np.array_equal(build_features(altered, 'bank9_path')[0], original_n)
    altered = {k: v.copy() for k, v in data.items()}
    for key in altered: altered[key][:, 10:] += 2
    after_n, after_f = build_features(altered, 'bank9_path')
    assert np.array_equal(after_n[:, :10], original_n[:, :10]) and np.array_equal(after_f[:, :10], original_f[:, :10])


def existing_audit():
    rows = []
    score = pd.read_csv(V2/'supplement_exact/results/prediction_seed_means.csv')
    for world in ['maxwell', 'boucwen']:
        for seed in range(5):
            for method in ['modular', 'analytic_flex']:
                path = V2/f'results/models/{world}/r{seed}/b24/clean/{method}/metadata.json'
                md = json.loads(path.read_text()); sc = score[(score.world == world)&(score.seed == seed)&(score.method == method)&(score.noise == 'clean')].iloc[0]
                rows.append(dict(world=world, seed=seed, method=method, val_normal_rmse=md['selected']['normal_validation_rmse'],
                                 val_friction_rmse=md['selected']['friction_validation_rmse'],
                                 old_test_conditional_normal_rmse=float(sc.conditional_normal_rmse),
                                 old_test_conditional_friction_rmse=float(sc.conditional_friction_rmse),
                                 old_test_open_position_rmse=float(sc.position_rmse),
                                 selected_update=md['selected'].get('best_update'), seconds=md['seconds']))
    pd.DataFrame(rows).to_csv(OUT/'existing_five_seed_audit.csv', index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True); torch.set_num_threads(2)
    test_causality()
    paths = [V2/f'results/data/{world}/r0/{split}.{suffix}' for world in ['maxwell', 'boucwen'] for split in ['train', 'val'] for suffix in ['npz', 'json']]
    declaration = dict(exploratory=True, worlds=['maxwell', 'boucwen'], dataset_replicate=0, model_seed=0,
                       variants=VARIANTS, history_taus_old=TAUS_OLD, history_taus_dense=TAUS_LONG, updates=UPDATES,
                       optimizer='Adam lr .002 first6000 updates, .0005 remaining6000; wd1e-6; clip5; batch512 per head',
                       selection='minimum isolated val scaled loss every200 updates plus1200 and12000; no fresh-test claim',
                       ridge='complete normal monomials degree<=3; friction degree<=2 plus absolute inputs; train-only scaling',
                       ridge_lambdas=LAMBDAS, parameter_information='public measured paths only; current/past N excluded from normal head',
                       no_new_test_generated=True, source_sha256={str(path): sha(path) for path in paths}, script_sha256=sha(__file__))
    manifest = OUT/'declaration.json'
    if manifest.exists(): assert json.loads(manifest.read_text()) == declaration, 'diagnostic declaration changed; preserve old results'
    else: dump(manifest, declaration)
    existing_audit(); results = []
    for world in declaration['worlds']:
        td, tr = load_data(world, 'train'); vd, vr = load_data(world, 'val')
        for variant in VARIANTS:
            train, val = local_data(td, tr, variant['feature']), local_data(vd, vr, variant['feature'])
            results.append(fit_mlp(world, variant, train, val))
            if variant['name'] != 'bank9_path_wide': results.append(fit_ridge(world, variant['feature'], train, val))
    dump(OUT/'results.json', results)
    flat = [{k: row[k] for k in ['world', 'kind', 'variant', 'feature', 'parameters', 'seconds']} |
            {split+'_'+metric: row[split][metric] for split in ['train', 'val'] for metric in ['normal_rmse', 'friction_rmse', 'scaled_loss']}
            for row in results]
    pd.DataFrame(flat).to_csv(OUT/'summary.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    for wi, world in enumerate(declaration['worlds']):
        selection = [r for r in results if r['world'] == world]
        for hi, head in enumerate(['normal', 'friction']):
            ax = axes[wi, hi]; ax.bar(np.arange(len(selection)), [r['val'][head+'_rmse'] for r in selection], color=['#377c9d' if r['kind']=='ridge' else '#b9483f' for r in selection])
            ax.set(xticks=np.arange(len(selection)), xticklabels=[r['variant'] for r in selection], title=world+' · '+head, ylabel='Isolierter Validierungs-RMSE (N)', yscale='log')
            plt.setp(ax.get_xticklabels(), rotation=36, ha='right')
    fig.savefig(OUT/'local_validation.png', dpi=170, bbox_inches='tight'); plt.close(fig)
    print('LOCAL_DIAGNOSTICS_COMPLETE', OUT/'summary.csv', flush=True)


if __name__ == '__main__': main()

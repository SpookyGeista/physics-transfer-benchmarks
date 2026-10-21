"""Train frozen v2 models exclusively on isolated calibration/validation files."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from decision_study.common import CFG, ROOT, digest, dump, features, history_init, history_update
from decision_study.models import Neural, fit_classical, freeze, parameter_count


def measured_arrays(arrays, noise, seed, split):
    """Rebuild all histories from causal measured signals, with common RNG per method."""
    if noise == 'clean':
        return np.asarray(arrays['features']), np.asarray(arrays['normal']), np.asarray(arrays['friction'])
    rng = np.random.default_rng(7200000 + seed*100 + {'train': 1, 'val': 2, 'privileged': 3}[split])
    sigma = CFG['noise']
    d = np.asarray(arrays['delta']) + rng.normal(0, sigma['delta'], arrays['delta'].shape)
    r = np.asarray(arrays['compression_rate'] if 'compression_rate' in arrays else arrays['rate']) + rng.normal(0, sigma['rate'], arrays['rate'].shape)
    v = np.asarray(arrays['state'])[..., 1] + rng.normal(0, sigma['velocity'], d.shape)
    n = np.asarray(arrays['normal']) + rng.normal(0, sigma['normal'], d.shape)
    ft = np.asarray(arrays['friction']) + rng.normal(0, sigma['friction'], d.shape)
    h = history_init(d[:, 0], v[:, 0], n[:, 0])
    f = np.empty((*d.shape, 13), dtype=np.float32)
    for t in range(d.shape[1]):
        f[:, t] = features(d[:, t], r[:, t], v[:, t], h, n[:, t])
        h = history_update(h, d[:, t], v[:, t], n[:, t])
    return f, n, ft


def dataset(data_root, world, seed, split, noise, budget=None):
    if split not in ('train', 'val', 'privileged'):
        raise ValueError('training cannot load evaluation splits')
    path = Path(data_root)/world/f'r{seed}'/f'{split}.npz'
    with np.load(path) as arrays:
        f, n, ft = measured_arrays(arrays, noise, seed, split)
    rows = json.loads(path.with_suffix('.json').read_text())['rows']
    assert len(rows) == len(f)
    ids = [i for i, row in enumerate(rows) if budget is None or row['profile'] < budget]
    if split != 'privileged':
        assert all(rows[i]['mode'] in ('A', 'B') for i in ids), 'isolated split has combined supervision'
    f, n, ft = f[ids], n[ids], ft[ids]
    mn = np.repeat([rows[i]['mode'] in ('B', 'AB') for i in ids], f.shape[1])
    mf = np.repeat([rows[i]['mode'] in ('A', 'AB') for i in ids], f.shape[1])
    return (f.reshape(-1, 13), n.ravel(), ft.ravel(), mn, mf), [rows[i]['id'] for i in ids], {
        str(path): digest(path), str(path.with_suffix('.json')): digest(path.with_suffix('.json'))}


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def train_one(method, train, validation, updates, dest, fingerprint, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tic = time.perf_counter()
    f, n, ft, mn, mf = train
    if method in ('nominal', 'identified', 'analytic_v1', 'analytic_flex'):
        model = fit_classical(method, f, n, ft, mn, mf, validation)
        dump(dest/'classical.json', dict(kind=model.kind, params=model.params))
        return model, [], {}, time.perf_counter()-tic
    kind = 'joint' if method == 'privileged' else method
    model = Neural(kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=1e-6)
    f, n, ft, mn, mf = [torch.as_tensor(x) for x in train]
    vf, vn, vft, vmn, vmf = [torch.as_tensor(x) for x in validation]
    idx_n, idx_f = torch.where(mn)[0], torch.where(mf)[0]
    curves, best, best_state, best_epoch, first, saved_seconds = [], float('inf'), None, -1, 0, 0.
    resume_path = dest/'training_state.pt'
    if resume_path.exists():
        saved = torch.load(resume_path, map_location='cpu', weights_only=True)
        assert saved['fingerprint'] == fingerprint, f'stale interrupted run: {dest}'
        model.load_state_dict(saved['state']); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        curves, best, best_state, best_epoch = saved['curves'], saved['best'], saved['best_state'], saved['best_epoch']
        first, saved_seconds = saved['next_update'], saved['seconds']
    for update in range(first, updates):
        half = CFG['batch_size']//2
        bn = idx_n[torch.randint(len(idx_n), (half,))]
        bf = idx_f[torch.randint(len(idx_f), (half,))]
        loss = ((model.normal(f[bn])-n[bn])/20).square().mean() + ((model.friction(f[bf])-ft[bf])/10).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite training loss: {dest}, update {update}')
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if update % 50 == 0 or update == updates-1:
            with torch.no_grad():
                vl = (((model.normal(vf[vmn])-vn[vmn])/20).square().mean() +
                      ((model.friction(vf[vmf])-vft[vmf])/10).square().mean()).item()
            curves.append(dict(update=update, train_loss=float(loss.item()), val_loss=vl))
            if vl < best:
                best, best_state, best_epoch = vl, copy.deepcopy(model.state_dict()), update
            atomic_torch_save(dict(kind=kind, fingerprint=fingerprint, state=model.state_dict(),
                                   optimizer=optimizer.state_dict(), rng=torch.get_rng_state(),
                                   curves=curves, best=best, best_state=best_state, best_epoch=best_epoch,
                                   next_update=update+1, seconds=saved_seconds+time.perf_counter()-tic), resume_path)
    assert best_state is not None
    model.load_state_dict(best_state); freeze(model)
    atomic_torch_save(dict(kind=kind, state=model.state_dict()), dest/'model.pt')
    dump(dest/'learning.json', curves)
    return model, curves, dict(best_update=best_epoch, validation_loss=best), saved_seconds+time.perf_counter()-tic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(ROOT/'results/data'))
    parser.add_argument('--out', default=str(ROOT/'results/models'))
    parser.add_argument('--worlds', nargs='+', default=CFG['worlds'])
    parser.add_argument('--seeds', type=int, nargs='+', default=CFG['seeds'])
    parser.add_argument('--budgets', type=int, nargs='+', default=CFG['budgets'])
    parser.add_argument('--noise', nargs='+', choices=['clean', 'noisy'], default=CFG['noise_conditions'])
    parser.add_argument('--updates', type=int, default=CFG['updates'])
    parser.add_argument('--methods', nargs='+', default=[m for m in CFG['methods'] if m != 'decoupled'])
    args = parser.parse_args()
    assert args.updates > 0
    if 'decoupled' in args.methods:
        parser.error('decoupled reuses modular checkpoints; it is an evaluation ablation')
    torch.set_num_threads(2)
    code = {str(path): digest(path) for path in [ROOT/'configs/experiment.json',
            ROOT/'src/decision_study/common.py', ROOT/'src/decision_study/models.py', Path(__file__).resolve()]}
    for world in args.worlds:
        for seed in args.seeds:
            for noise in args.noise:
                validation, val_ids, val_hashes = dataset(args.data, world, seed, 'val', noise)
                for budget in args.budgets:
                    base, base_ids, base_hashes = dataset(args.data, world, seed, 'train', noise, budget)
                    for method in args.methods:
                        dest = Path(args.out)/world/f'r{seed}'/f'b{budget}'/noise/method
                        dest.mkdir(parents=True, exist_ok=True)
                        train, ids, sources = base, base_ids.copy(), dict(base_hashes)
                        if method == 'privileged':
                            extra, extra_ids, extra_hashes = dataset(args.data, world, seed, 'privileged', noise)
                            train = tuple(np.concatenate((a, b)) for a, b in zip(base, extra))
                            ids.extend(extra_ids); sources.update(extra_hashes)
                        spec = dict(world=world, seed=seed, budget=budget, noise=noise, method=method,
                                    updates=args.updates, code=code, training_sources=sources,
                                    validation_sources=val_hashes, config=CFG)
                        fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
                        if (dest/'metadata.json').exists():
                            md = json.loads((dest/'metadata.json').read_text())
                            assert md['fingerprint'] == fingerprint, f'completed run inputs/args changed: {dest}; use fresh --out'
                            for path, sha in md['model_hashes'].items():
                                assert digest(path) == sha, f'checkpoint changed: {path}'
                            print('SKIP verified', dest, flush=True)
                            continue
                        model, curves, selected, elapsed = train_one(method, train, validation, args.updates,
                                                                    dest, fingerprint, seed)
                        vf, vn, vft, vmn, vmf = validation
                        pn, pft = model.predict_local(vf)
                        selected.update(normal_validation_rmse=float(np.sqrt(np.mean((pn[vmn]-vn[vmn])**2))),
                                        friction_validation_rmse=float(np.sqrt(np.mean((pft[vmf]-vft[vmf])**2))))
                        if method == 'analytic_flex':
                            selected['ridge_lambda'] = {head: model.params[head]['selected_lambda'] for head in ('normal', 'friction')}
                        model_files = list(dest.glob('classical.json')) + list(dest.glob('model.pt'))
                        dump(dest/'metadata.json', dict(**spec, fingerprint=fingerprint, training_ids=ids,
                             validation_ids=val_ids, parameters=parameter_count(model), seconds=elapsed,
                             selected=selected, optimizer_updates=args.updates if curves else 0,
                             frozen=True, test_data_used=False, device='cpu', torch_threads=2,
                             noise_rng_rule='7200000 + replicate*100 + split(1 train,2 val,3 privileged)',
                             max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             model_hashes={str(path): digest(path) for path in model_files}))
                        print('DONE', world, seed, budget, noise, method, f'{elapsed:.2f}s', selected, flush=True)


if __name__ == '__main__':
    main()

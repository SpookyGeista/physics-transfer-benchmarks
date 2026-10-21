"""Positive separable least squares using only local measured v/N/Ft.

No truth module, parameters, latent labels or task data are read here. Three
nonlinear alpha=mu/k plus three conditional linear k are six fitted parameters.
Only training loss selects starts; validation is first loaded after model save.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from lock import ROOT, digest, verify
from model import PIModel, basis


class BudgetExpired(RuntimeError):
    pass


def positive_readout(matrix, target, alpha, config):
    klo, khi = config['stiffness_bounds_N_m']
    mlo, mhi = config['capacity_bounds']
    lower = np.maximum(klo, mlo/alpha)
    upper = np.minimum(khi, mhi/alpha)
    k = np.linalg.lstsq(matrix, target, rcond=None)[0]
    if np.any(k < lower) or np.any(k > upper):
        solved = lsq_linear(matrix, target, bounds=(lower, upper),
                            method='bvls', tol=1e-10, max_iter=100)
        if not solved.success:
            raise RuntimeError('conditional positive least squares failed')
        k = solved.x
    return k


def fit_measured(velocity, normal, target, dt, fit_config, rng_seed):
    """Fit from three sensor arrays. Known relaxed episode starts are a prior."""
    start = time.monotonic()
    v, N, y = [np.asarray(a, dtype=float) for a in (velocity, normal, target)]
    if v.ndim != 2 or N.shape != v.shape or y.shape != v.shape:
        raise ValueError('identical (episodes,steps) sensor arrays required')
    if not all(np.isfinite(a).all() for a in (v, N, y)) or np.any(N < 0):
        raise ValueError('finite observations and nonnegative N required')
    target_flat = y.ravel()
    cfg = fit_config
    rng = np.random.default_rng(rng_seed)
    initial = [np.asarray(cfg['nominal_start_yield_at_15N_m'])/15.]
    lo, hi = np.log(np.asarray(cfg['random_start_yield_at_15N_m'])/15.)
    initial.extend(np.exp(np.sort(rng.uniform(lo, hi, 3))) for _ in range(cfg['multistarts']-1))
    bounds = np.log(np.asarray(cfg['yield_distance_at_15N_bounds_m'])/15.)
    best = dict(sse=np.inf)
    calls = 0
    starts = []
    def residual(log_alpha):
        nonlocal calls, best
        if time.monotonic()-start >= cfg['max_wall_seconds']:
            raise BudgetExpired()
        alpha = np.sort(np.exp(log_alpha))
        matrix = basis(alpha, v, N, dt)
        k = positive_readout(matrix, target_flat, alpha, cfg)
        errors = matrix @ k - target_flat
        sse = float(errors @ errors)
        calls += 1
        if sse < best['sse']:
            best = dict(sse=sse, alpha=alpha.copy(), k=k.copy(), evaluation=calls)
        return errors
    budget_expired = False
    for index, guess in enumerate(initial):
        try:
            result = least_squares(residual, np.log(guess), bounds=bounds,
                                   jac='2-point', x_scale='jac', max_nfev=cfg['max_nfev_per_start'],
                                   diff_step=cfg['diff_step'], ftol=cfg['ftol'],
                                   xtol=cfg['xtol'], gtol=cfg['gtol'])
            starts.append(dict(index=index, nfev=result.nfev, status=int(result.status),
                               success=bool(result.success), sse=float(result.fun @ result.fun),
                               message=result.message))
        except BudgetExpired:
            starts.append(dict(index=index, status='wall_budget_expired'))
            budget_expired = True
            break
    if not np.isfinite(best['sse']):
        raise RuntimeError('no finite fit inside budget')
    model = PIModel(best['k'], best['k']*best['alpha'])
    base_prediction = model.predict(v, N, dt).ravel()
    log_coeff = np.log(np.r_[model.k, model.mu])
    jacobian = np.empty((y.size, 6))
    for column in range(6):
        moved = log_coeff.copy(); moved[column] += 1e-5
        jacobian[:, column] = (PIModel(np.exp(moved[:3]), np.exp(moved[3:])).predict(v, N, dt).ravel()
                               - base_prediction)/1e-5
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.sum(singular > singular[0]*max(jacobian.shape)*np.finfo(float).eps))
    diagnostics = dict(training_rmse_N=float(np.sqrt(best['sse']/y.size)),
                       sse=best['sse'], calls=calls, best_evaluation=best['evaluation'],
                       multistarts=starts, wall_seconds=time.monotonic()-start,
                       budget_expired=budget_expired,
                       training_logparameter_jacobian_rank=rank,
                       training_logparameter_jacobian_singular_values=singular.tolist(),
                       training_logparameter_jacobian_condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None,
                       selection='training SSE only; no validation used')
    return model, diagnostics


def load_measured(path, manifest):
    relative = str(Path(path).relative_to(ROOT))
    if digest(path) != manifest['files'][relative]:
        raise RuntimeError(f'input hash mismatch: {relative}')
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'v','N','Ft','dt'}:
            raise ValueError('forbidden or missing data fields')
        return {key: data[key].copy() for key in data.files}


def _job(job):
    case, seed, condition = job
    source_lock = verify()
    cfg = json.loads((ROOT/'config.json').read_text())
    manifest_path = ROOT/'results/dataset_manifest.json'
    manifest_sha = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest['design_lock_sha256'] != source_lock:
        raise RuntimeError('dataset source lock differs')
    data_folder = ROOT/'results/data'/case/f'r{seed}'/condition
    out = ROOT/'results/models'/case/f'r{seed}'/condition
    if out.exists():
        raise RuntimeError(f'refusing overwrite: {out}')
    train_path = data_folder/'train.npz'
    train = load_measured(train_path, manifest)
    model, diagnostics = fit_measured(train['v'], train['N'], train['Ft'], float(train['dt']),
                                     cfg['fit'], cfg['fit_seed_base']+seed)
    out.mkdir(parents=True)
    model.metadata = dict(case=case, seed=seed, condition=condition,
                          design_lock_sha256=source_lock, dataset_manifest_sha256=manifest_sha,
                          training_npz_sha256=digest(train_path),
                          initial_preparation='known relaxed', fit=diagnostics)
    model.save(out/'model.json')
    # The selected coefficients have been saved before validation labels load.
    val_path = data_folder/'val.npz'
    val = load_measured(val_path, manifest)
    prediction = model.predict(val['v'], val['N'], float(val['dt']))
    residual = prediction-val['Ft']
    hold = abs(val['v']) < 1e-12
    np.savez_compressed(out/'validation_predictions.npz', prediction=prediction,
                        measured=val['Ft'], hold_mask=hold)
    diagnostics.update(validation_rmse_observed_N=float(np.sqrt(np.mean(residual**2))),
                       validation_hold_rmse_observed_N=float(np.sqrt(np.mean(residual[hold]**2))),
                       validation_npz_sha256=digest(val_path), model_sha256=digest(out/'model.json'),
                       prediction_sha256=digest(out/'validation_predictions.npz'), pid=os.getpid(),
                       design_lock_sha256=source_lock, dataset_manifest_sha256=manifest_sha)
    verify()
    if digest(manifest_path) != manifest_sha:
        raise RuntimeError('dataset manifest changed while fitting')
    if digest(train_path) != model.metadata['training_npz_sha256']:
        raise RuntimeError('training data changed while fitting')
    (out/'diagnostics.json').write_text(json.dumps(diagnostics, indent=2, allow_nan=False)+'\n')
    return dict(case=case, seed=seed, condition=condition,
                train_rmse_N=diagnostics['training_rmse_N'],
                val_rmse_N=diagnostics['validation_rmse_observed_N'],
                seconds=diagnostics['wall_seconds'], budget_expired=diagnostics['budget_expired'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    verify()
    cfg = json.loads((ROOT/'config.json').read_text())
    jobs = [(case, seed, condition) for case in cfg['cases']
            for seed in cfg['replicate_seeds'] for condition in cfg['conditions']]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(_job, jobs):
            print(json.dumps(result), flush=True)
    verify()
    dataset_manifest = json.loads((ROOT/'results/dataset_manifest.json').read_text())
    for relative, expected in dataset_manifest['files'].items():
        if digest(ROOT/relative) != expected:
            raise RuntimeError(f'data changed: {relative}')
    files = {str(path.relative_to(ROOT)):digest(path)
             for path in (ROOT/'results/models').rglob('*') if path.is_file()}
    (ROOT/'results/fit_manifest.json').write_text(json.dumps(dict(
        design_lock_sha256=verify(), dataset_manifest_sha256=digest(ROOT/'results/dataset_manifest.json'),
        fit_count=len(jobs), files=files), indent=2)+'\n')
    print(f'all {len(jobs)} fits completed in {time.monotonic()-started:.2f}s', flush=True)


if __name__ == '__main__':
    main()

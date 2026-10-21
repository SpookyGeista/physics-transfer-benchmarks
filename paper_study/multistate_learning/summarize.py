"""Post-selection validation against independent noisy and clean measurements.

This report never selects/refits a checkpoint. Clean labels for a noisy fit are
opened only here, after all completed coefficients/validation predictions exist.
"""
import csv
import json
from pathlib import Path

import numpy as np

from lock import ROOT, digest, verify


def main():
    source_lock = verify()
    cfg = json.loads((ROOT/'config.json').read_text())
    data_manifest = json.loads((ROOT/'results/dataset_manifest.json').read_text())
    fit_manifest = json.loads((ROOT/'results/fit_manifest.json').read_text())
    if fit_manifest['design_lock_sha256'] != source_lock or fit_manifest['fit_count'] != 24:
        raise RuntimeError('incompatible or incomplete fit manifest')
    if fit_manifest['dataset_manifest_sha256'] != digest(ROOT/'results/dataset_manifest.json'):
        raise RuntimeError('dataset manifest changed')
    for manifest in (data_manifest, fit_manifest):
        for relative, expected in manifest['files'].items():
            if digest(ROOT/relative) != expected:
                raise RuntimeError(f'input artifact changed: {relative}')
    rows = []
    for case in cfg['cases']:
        for seed in cfg['replicate_seeds']:
            clean_path = ROOT/'results/data'/case/f'r{seed}'/'clean/val.npz'
            with np.load(clean_path) as data:
                clean = data['Ft'].copy()
            for condition in cfg['conditions']:
                folder = ROOT/'results/models'/case/f'r{seed}'/condition
                model = json.loads((folder/'model.json').read_text())
                diagnostic = json.loads((folder/'diagnostics.json').read_text())
                if diagnostic['design_lock_sha256'] != source_lock:
                    raise RuntimeError('model source lock mismatch')
                if digest(folder/'model.json') != diagnostic['model_sha256']:
                    raise RuntimeError('model hash mismatch')
                if digest(folder/'validation_predictions.npz') != diagnostic['prediction_sha256']:
                    raise RuntimeError('prediction hash mismatch')
                with np.load(folder/'validation_predictions.npz') as data:
                    pred, hold = data['prediction'], data['hold_mask']
                error = pred-clean
                row = dict(case=case, seed=seed, condition=condition,
                           training_rmse_N=diagnostic['training_rmse_N'],
                           validation_rmse_observed_N=diagnostic['validation_rmse_observed_N'],
                           validation_rmse_clean_N=float(np.sqrt(np.mean(error**2))),
                           hold_rmse_clean_N=float(np.sqrt(np.mean(error[hold]**2))),
                           wall_seconds=diagnostic['wall_seconds'], budget_expired=diagnostic['budget_expired'])
                for i in range(3):
                    row[f'k{i+1}_N_m']=model['stiffness_N_m'][i]
                    row[f'mu{i+1}']=model['friction_capacity'][i]
                rows.append(row)
    if len(rows) != 24:
        raise RuntimeError('incomplete planned fit grid')
    out = ROOT/'results/summary'
    if out.exists():
        raise RuntimeError('refusing summary overwrite')
    out.mkdir(parents=True)
    with (out/'fits.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups = []
    for case in cfg['cases']:
        for condition in cfg['conditions']:
            selected = [r for r in rows if r['case']==case and r['condition']==condition]
            groups.append(dict(case=case, condition=condition, replicates=len(selected),
                               mean_val_clean_rmse_N=float(np.mean([r['validation_rmse_clean_N'] for r in selected])),
                               max_val_clean_rmse_N=max(r['validation_rmse_clean_N'] for r in selected),
                               mean_hold_clean_rmse_N=float(np.mean([r['hold_rmse_clean_N'] for r in selected])),
                               mean_val_observed_rmse_N=float(np.mean([r['validation_rmse_observed_N'] for r in selected])),
                               budget_expired=sum(r['budget_expired'] for r in selected)))
    (out/'groups.json').write_text(json.dumps(groups, indent=2, allow_nan=False)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(design_lock_sha256=source_lock,
        dataset_manifest_sha256=digest(ROOT/'results/dataset_manifest.json'),
        fit_manifest_sha256=digest(ROOT/'results/fit_manifest.json'),
        fit_count=len(rows), coefficient_count_per_fit=6, model_selection='train only',
        files={p.name:digest(p) for p in out.iterdir() if p.is_file()}), indent=2)+'\n')
    verify()
    print(json.dumps(groups, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Locked r1 native MuJoCo/refinement audit; synthetic self-test before release.

Real-data mode requires complete source/model/test locks before opening any
holdout or prediction artifact. It never runs or refits a learner at finer dt.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from round1_lock import verify_code_lock, verify_model_lock, verify_test_lock
from round1_mujoco_reference import replay, make_model, compression, preparation

NUMERICAL_KEYS = (
    'native_velocity_tracking_max', 'native_step_position_residual_max',
    'mujoco_vs_physx_position_rmse_max', 'mujoco_vs_physx_position_absolute_max',
    'refinement_h1_h4_position_rmse_max', 'refinement_median_D24_D12_max',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths = [Path(__file__), ROOT/'src/round1_mujoco_reference.py',
             ROOT/'src/round1_lock.py', ROOT/'configs/round1.json']
    return {str(p.resolve()): digest(p) for p in paths}


def write_json_exclusive(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def position_errors(a, b):
    if np.shape(a) != np.shape(b):
        raise ValueError('Matching state arrays required; no implicit broadcasting')
    delta = np.asarray(a)-np.asarray(b)
    if delta.ndim != 3 or delta.shape[-1] != 2 or not np.isfinite(delta).all():
        raise ValueError('Finite matching [episode,time,(x,v)] arrays required')
    return dict(position_rmse_m=float(np.sqrt(np.mean(delta[..., 0]**2))),
                position_absolute_max_m=float(np.max(np.abs(delta[..., 0]))),
                velocity_rmse_m_s=float(np.sqrt(np.mean(delta[..., 1]**2))),
                velocity_absolute_max_m_s=float(np.max(np.abs(delta[..., 1]))),
                episode_position_rmse_m=np.sqrt(np.mean(delta[..., 0]**2, axis=1)).tolist())


def refinement(h1, h2, h4):
    if (np.shape(h1) != np.shape(h2) or np.shape(h1) != np.shape(h4)
            or np.ndim(h1) != 3 or np.shape(h1)[-1] != 2
            or not all(np.isfinite(a).all() for a in (h1, h2, h4))):
        raise ValueError('Finite, matching [episode,time,(x,v)] refinement arrays required')
    d12 = np.sqrt(np.mean((h1[..., 0]-h2[..., 0])**2, axis=1))
    d24 = np.sqrt(np.mean((h2[..., 0]-h4[..., 0])**2, axis=1))
    d14 = np.sqrt(np.mean((h1[..., 0]-h4[..., 0])**2, axis=1))
    # Exactly identical coarse solutions have ratio zero iff both differences
    # are zero. Infinite/overflow ratios are reported by a declared 1e100
    # sentinel, which preserves threshold failure and serializes finite JSON.
    with np.errstate(over='ignore', divide='ignore'):
        ratio = np.divide(d24, d12, out=np.where(d24 == 0, 0., 1e100), where=d12 != 0)
    ratio = np.minimum(ratio, 1e100)
    return dict(D12_episode_rmse_m=d12.tolist(), D24_episode_rmse_m=d24.tolist(),
                D14_episode_rmse_m=d14.tolist(), D24_D12_episode=ratio.tolist(),
                median_D24_D12=float(np.median(ratio)),
                h1_h4_position_rmse_m=float(np.sqrt(np.mean(d14**2))))


def verify_record(path, cfg, locks, normal_world, friction_world):
    """Called only after model/source lock verification by the CLI."""
    metadata_path = path.with_suffix('.json')
    meta = json.loads(metadata_path.read_text())
    if meta['locks'] != locks or meta['npz_sha256'] != digest(path):
        raise RuntimeError(f'Test lock or NPZ digest mismatch: {path}')
    if (meta['normal_world'], meta['friction_world'], meta['replicate']) != (normal_world, friction_world, 1):
        raise ValueError(f'Wrong reserved composition/replicate: {path}')
    for name, expected in meta['source_sha256'].items():
        if digest(name) != expected:
            raise RuntimeError(f'Test source changed: {name}')
    with np.load(path, allow_pickle=False) as saved:
        data = {k: saved[k] for k in saved.files}
    n, steps, warm_steps = 16, round(cfg['motion_seconds']/cfg['dt']), round(cfg['warmup_seconds']/cfg['dt'])
    if len(meta['rows']) != n or [r['profile'] for r in meta['rows']] != list(range(n)):
        raise ValueError('All sixteen original profiles in order are required')
    if (meta['dt'], meta['mass'], meta['steps'], meta['warm_steps']) != (cfg['dt'], cfg['mass'], steps, warm_steps):
        raise ValueError('Recorded duration, mass or observation grid differs')
    for name in ('state', 'next_state'):
        if data[name].shape != (n, steps, 2):
            raise ValueError(f'Wrong {name} shape')
    for name in ('u', 'd', 'rate', 'normal', 'friction'):
        if data[name].shape != (n, steps):
            raise ValueError(f'Wrong {name} shape')
    for name in ('warm_delta', 'warm_rate', 'warm_normal', 'warm_friction'):
        if data[name].shape != (n, warm_steps):
            raise ValueError(f'Wrong {name} shape')
    if not all(np.isfinite(a).all() for a in data.values()):
        raise FloatingPointError('Nonfinite native data; preserve result, numerical claim invalid')
    if np.max(np.abs(data['state'][:, 0])) > 1e-9 or np.max(np.abs(data['warm_state'])) > 1e-9:
        raise ValueError('Known zero-motion full prefix is required')
    for j in range(steps):
        dd, rr = compression(j*cfg['dt'], meta['rows'])
        np.testing.assert_allclose(dd, data['d'][:, j], atol=1e-14, rtol=0)
        np.testing.assert_allclose(rr, data['rate'][:, j], atol=1e-14, rtol=0)
    for j in range(warm_steps):
        dd, rr = preparation(j*cfg['dt'], meta['rows'])
        np.testing.assert_allclose(dd, data['warm_delta'][:, j], atol=1e-14, rtol=0)
        np.testing.assert_allclose(rr, data['warm_rate'][:, j], atol=1e-14, rtol=0)
    return data, meta


def verify_predictions(root, cfg, normal, friction, model_lock, test_digest, rows):
    predictions, consumed = {}, {}
    for method in cfg['methods']:
        path = root/f'{normal}_{friction}/r1/clean/{method}.npz'
        meta_path = path.with_suffix('.json')
        meta = json.loads(meta_path.read_text())
        if (meta['model_lock_sha256'] != model_lock or meta['test_sha256'] != test_digest
                or meta['npz_sha256'] != digest(path) or meta['rows'] != rows):
            raise RuntimeError(f'Prediction provenance mismatch: {path}')
        if (meta['normal_world'], meta['friction_world'], meta['seed'], meta['noise'], meta['method']) != (normal, friction, 1, 'clean', method):
            raise ValueError(f'Wrong prediction cell: {path}')
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in ('state', 'normal', 'friction', 'guard')}
        shape = (16, round(cfg['motion_seconds']/cfg['dt']))
        if arrays['state'].shape != (*shape, 2) or any(arrays[k].shape != shape for k in ('normal', 'friction', 'guard')):
            raise ValueError('Prediction shapes differ from fixed coarse observation grid')
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise FloatingPointError(f'Nonfinite prediction, never replace or omit: {path}')
        predictions[method] = arrays
        consumed.update({str(path.resolve()): digest(path), str(meta_path.resolve()): digest(meta_path)})
    return predictions, consumed


def save_replay(path, arrays, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    write_json_exclusive(path.with_suffix('.json'), dict(meta, npz_sha256=digest(path)))


def load_or_run(path, normal, friction, rows, u, substeps, cfg, provenance, deadline):
    meta_path = path.with_suffix('.json')
    if path.exists() or meta_path.exists():
        if not path.exists() or not meta_path.exists():
            raise RuntimeError(f'Partial artifact retained; explicit review required: {path}')
        old = json.loads(meta_path.read_text())
        if old['provenance'] != provenance or old['npz_sha256'] != digest(path) or old['substeps'] != substeps:
            raise RuntimeError(f'Existing independent replay belongs to different inputs: {path}')
        with np.load(path, allow_pickle=False) as saved:
            return {k: saved[k] for k in saved.files}, old
    start = time.monotonic()
    arrays, info = replay(normal, friction, rows, u, substeps=substeps, dt=cfg['dt'],
                          warmup_seconds=cfg['warmup_seconds'], deadline=deadline)
    info.update(normal_world=normal, friction_world=friction, replicate=1, rows=rows,
                provenance=provenance, seconds=time.monotonic()-start)
    save_replay(path, arrays, info)
    print(f'MUJOCO {normal}/{friction} h{substeps}: {info["seconds"]:.3f} s', flush=True)
    return arrays, info


def report(summary):
    text = ['# Unabhängige Round-1-Backend- und Zeitschrittkontrolle', '',
            'Vollständig: native MuJoCo-Schlitten mit unabhängig codierten, unveränderten synthetischen Materialgesetzen. '
            'Die reservierte saubere Replikation r1 umfasst sämtliche vier Kombinationen und 16 Profile. '
            'Dies prüft Implementierung und Diskretisierung bei einem Seed; es ist keine unabhängige empirische Materialvalidierung.', '',
            'Die vier Sekunden Präparation werden für h=2,5/1,25/0,625 ms jeweils neu integriert. '
            'Anschließend bleiben die originalen Kraftkommandos pro grobem Intervall konstant; Kompressionswelle und Rate '
            'werden analytisch am feinen Raster ausgewertet. Kompression und geometrische Gegenkraft verwenden den tatsächlichen '
            'MuJoCo-Zustand. Es gibt nach Freigabe keine inverse Kraftkorrektur und keine gemessene Zustandseinspeisung.', '',
            '| feste Numerikgröße | beobachtetes Maximum über vier Zellen | Grenze | bestanden |',
            '|---|---:|---:|---|']
    for k, value in summary['observed'].items():
        text.append(f'| {k} | {value:.9g} | {summary["limits"][k]:.9g} | {value <= summary["limits"][k]} |')
    text += ['', f'Gesamtes Numerik-Gate: **{"bestanden" if summary["passed"] else "nicht bestanden"}**. '
             'Die Schwellen werden je Komposition angewendet; die publizierten Gesamtwerte sind jeweils die ungünstigste Zelle. '
             'Der Verfeinerungsquotient ist pro Profil definiert und anschließend innerhalb der 16 Profile einer Zelle medianisiert.', '',
             '| Normal / Reibung | h1–PhysX RMSE mm | h1–PhysX max mm | h1–h4 RMSE mm | Median D24/D12 |',
             '|---|---:|---:|---:|---:|']
    for cell in summary['cells']:
        e, r = cell['h1_vs_physx'], cell['refinement']
        text.append(f'| {cell["normal_world"]} / {cell["friction_world"]} | {1000*e["position_rmse_m"]:.6g} | '
                    f'{1000*e["position_absolute_max_m"]:.6g} | {1000*r["h1_h4_position_rmse_m"]:.6g} | {r["median_D24_D12"]:.6g} |')
    text += ['', '## Unveränderte grobe Modellprognosen gegen feinere Referenz', '',
             'Kein Modell wird bei feinerem h erneut ausgeführt oder trainiert. Insbesondere bleibt der diskrete GRU bei seinem '
             'Trainingsraster. Dieselben eingefrorenen, vollständig offenen 480 Prognosepunkte werden mit zeitlich ausgerichteten '
             'feineren nativen Referenzzuständen verglichen. Die folgenden Rangfolgen sind deskriptive Backenddiagnostik; '
             'sie ersetzen keine primären acht-Seed-Kontraste.', '',
             '| Normal / Reibung | Referenz | Methoden nach mittlerer Profil-RMSE mm |', '|---|---|---|']
    for cell in summary['cells']:
        for truth, ranking in cell['prediction_rankings'].items():
            rendered = ', '.join(f'{r["method"]}: {1000*r["mean_episode_position_rmse_m"]:.4g}' for r in ranking)
            text.append(f'| {cell["normal_world"]} / {cell["friction_world"]} | {truth} | {rendered} |')
    text += ['', 'Alle 64 Profilwerte, D12/D24/D14, sämtliche Methodenfehler, Prognoseguards und Provenienzhashes stehen '
             'in `results/round1/mujoco/refinement_summary.json`; rohe native Verläufe liegen daneben. '
             'Die maschinenlesbare Entscheidungsevidenz ist `results/round1/numerical_validity.json`.', '',
             'Die synthetischen Vorprüfungen und Fortsetzungsbefehle sind in der Quellenhistorie und diesem ausführbaren Test erhalten:', '',
             '```bash', 'OPENBLAS_NUM_THREADS=1 /home/loya2/miniconda3/envs/mujoco/bin/python research_next/tests/test_round1_backend.py',
             'OPENBLAS_NUM_THREADS=1 /home/loya2/miniconda3/envs/mujoco/bin/python research_next/scripts/check_round1_backends.py', '```', '']
    return '\n'.join(text)


def run(args):
    # All three guards precede every holdout/prediction read, including metadata.
    # Native metadata predates the test lock; its nested `locks` therefore keeps
    # the original code/model pair, with test provenance recorded separately.
    locks = dict(code_lock_sha256=verify_code_lock(), model_lock_sha256=verify_model_lock())
    test_lock = verify_test_lock()
    cfg = json.loads((ROOT/'configs/round1.json').read_text())
    if cfg['dt'] != .0025 or cfg['mass'] != 2 or cfg['warmup_seconds'] != 4 or cfg['motion_seconds'] != 1.2:
        raise ValueError('Independent implementation requires the declared round-1 grid and mechanics')
    source_hashes = sources()
    deadline = time.monotonic()+cfg['resources']['backend_wall_seconds_limit']
    consumed, cells = {}, []
    observed = dict.fromkeys(NUMERICAL_KEYS, 0.)
    for normal, friction in cfg['compositions']:
        path = args.input/f'{normal}_{friction}/r1.npz'
        recorded, metadata = verify_record(path, cfg, locks, normal, friction)
        predictions, prediction_hashes = verify_predictions(args.predictions, cfg, normal, friction,
            locks['model_lock_sha256'], digest(path), metadata['rows'])
        inputs = {str(path.resolve()): digest(path), str(path.with_suffix('.json').resolve()): digest(path.with_suffix('.json'))}
        consumed.update(inputs)
        consumed.update(prediction_hashes)
        provenance = dict(locks=locks, test_lock_sha256=test_lock,
                          source_sha256=source_hashes, input_sha256=inputs)
        native, infos = {}, {}
        for substeps in (1, 2, 4):
            output_path = args.out/f'{normal}_{friction}_r1_h{substeps}.npz'
            native[substeps], infos[substeps] = load_or_run(output_path, normal, friction, metadata['rows'],
                recorded['u'], substeps, cfg, provenance, deadline)
            consumed[str(output_path.resolve())] = digest(output_path)
            consumed[str(output_path.with_suffix('.json').resolve())] = digest(output_path.with_suffix('.json'))
        h1_error = position_errors(native[1]['next_state'], recorded['next_state'])
        refined = refinement(*[native[h]['next_state'] for h in (1, 2, 4)])
        # Recompute the two native gates directly from immutable arrays.
        tracking = float(np.max(np.abs(recorded['next_state'][..., 1]-recorded['target_velocity'][:, 1:])))
        residual = float(np.max(np.abs(recorded['next_state'][..., 0]-recorded['state'][..., 0]-cfg['dt']*recorded['next_state'][..., 1])))
        for key, value in [('native_velocity_tracking_max', tracking), ('native_step_position_residual_max', residual)]:
            np.testing.assert_allclose(value, metadata['numerical'][key], atol=1e-15, rtol=0)
        measurements = dict(native_velocity_tracking_max=tracking, native_step_position_residual_max=residual,
            mujoco_vs_physx_position_rmse_max=h1_error['position_rmse_m'],
            mujoco_vs_physx_position_absolute_max=h1_error['position_absolute_max_m'],
            refinement_h1_h4_position_rmse_max=refined['h1_h4_position_rmse_m'],
            refinement_median_D24_D12_max=refined['median_D24_D12'])
        for k, value in measurements.items():
            observed[k] = max(observed[k], value)
        errors, rankings = {}, {}
        for name, truth in [('PhysX_h1', recorded), *[(f'MuJoCo_h{h}', native[h]) for h in (1, 2, 4)]]:
            errors[name] = {}
            ranking = []
            for method, predicted in predictions.items():
                metrics = position_errors(predicted['state'], truth['next_state'])
                metrics.update(normal_rmse_N=float(np.sqrt(np.mean((predicted['normal']-truth['normal'])**2))),
                               friction_rmse_N=float(np.sqrt(np.mean((predicted['friction']-truth['friction'])**2))),
                               guard_steps=int(np.sum(predicted['guard'])),
                               guard_episodes=int(np.sum(np.any(predicted['guard'], axis=1))))
                errors[name][method] = metrics
                ranking.append(dict(method=method, mean_episode_position_rmse_m=float(np.mean(metrics['episode_position_rmse_m']))))
            rankings[name] = sorted(ranking, key=lambda r: r['mean_episode_position_rmse_m'])
        force_parity = {key: dict(rmse_N=float(np.sqrt(np.mean((native[1][key]-recorded[key])**2))),
                                 absolute_max_N=float(np.max(np.abs(native[1][key]-recorded[key]))))
                        for key in ('normal', 'friction', 'warm_normal', 'warm_friction')}
        cells.append(dict(normal_world=normal, friction_world=friction, replicate=1, rows=metadata['rows'],
            h1_vs_physx=h1_error, h1_force_parity=force_parity, refinement=refined,
            observed=measurements, prediction_errors=errors, prediction_rankings=rankings,
            warm_motion_max=max(info['warm_motion_max'] for info in infos.values())))
    if len(cells) != 4 or sum(len(c['rows']) for c in cells) != 64:
        raise RuntimeError('All four Cartesian cells and 64 profiles are required')
    if (verify_test_lock() != test_lock or verify_model_lock() != locks['model_lock_sha256']
            or verify_code_lock() != locks['code_lock_sha256'] or sources() != source_hashes):
        raise RuntimeError('Source/model/test lock changed during backend audit')
    for path, expected in consumed.items():
        if digest(path) != expected:
            raise RuntimeError(f'Consumed artifact changed during audit: {path}')
    limits = {key: cfg['numerical_validity'][key] for key in NUMERICAL_KEYS}
    summary = dict(complete=True, scope='r1 clean, all four Cartesian cells and sixteen profiles each',
                   observed=observed, limits=limits, passed=all(observed[k] <= limits[k] for k in NUMERICAL_KEYS),
                   aggregation='thresholds per composition; worst of four cell metrics; per-cell median of sixteen D24/D12 ratios',
                   cells=cells, source_sha256=source_hashes, consumed_sha256=consumed, locks=locks,
                   test_lock_sha256=test_lock,
                   generated_utc=datetime.now(timezone.utc).isoformat(),
                   fixed_prediction_notice='Same coarse model trajectories scored against finer native truth; no learner rerun/refit')
    summary_path = args.out/'refinement_summary.json'
    evidence_path = args.out.parent/'numerical_validity.json'
    if summary_path.exists() or evidence_path.exists():
        raise RuntimeError('Final numerical evidence already exists; preserve it and review explicitly')
    write_json_exclusive(summary_path, summary)
    write_json_exclusive(evidence_path, dict(complete=True, observed=observed, source_sha256=source_hashes,
        locks=locks, test_lock_sha256=test_lock, scope=summary['scope'], aggregation=summary['aggregation'],
        detailed_summary=str(summary_path.resolve()), detailed_summary_sha256=digest(summary_path),
        passed=summary['passed'], consumed_sha256=consumed))
    (ROOT/'reports/round1_backend.md').write_text(report(summary))
    print(json.dumps(dict(complete=True, passed=summary['passed'], observed=observed), indent=2), flush=True)


def self_test():
    """Small native dynamics check, with no reserved seeds or experiment reads."""
    import mujoco
    model = make_model(2, .0025)
    data = mujoco.MjData(model)
    for _ in range(20):
        data.qfrc_applied[:] = [2., -4.]
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.qvel, [.05, -.1], atol=1e-14)
    print(json.dumps(dict(synthetic_only=True, native_backend=mujoco.__version__,
                         known_force_final_velocity=data.qvel.tolist(),
                         full_tests='research_next/tests/test_round1_backend.py'), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'results/round1/test')
    parser.add_argument('--predictions', type=Path, default=ROOT/'results/round1/predictions')
    parser.add_argument('--out', type=Path, default=ROOT/'results/round1/mujoco')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        try:
            run(args)
        except Exception as error:
            # Partial native replays remain untouched; absent complete evidence
            # keeps the primary analysis pending rather than silently reduced.
            print(f'INCOMPLETE BACKEND AUDIT: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
            raise


if __name__ == '__main__':
    main()

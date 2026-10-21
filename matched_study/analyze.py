"""Descriptive matched-law analysis; no simulator, fitting or policy selection."""
from pathlib import Path
import argparse
import itertools
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/physics-transfer-matched-analysis-mpl')
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CHANNELS = ('position_m', 'velocity_m_s', 'normal_N', 'friction_N')
METHODS = ('prior', 'posterior', 'mle_start1', 'mle_two_start', 'privileged_history')
WORK = ('prep_signed_tangential_work_J', 'prep_absolute_tangential_work_J',
        'prep_signed_normal_work_J', 'prep_absolute_normal_work_J',
        'future_signed_tangential_work_J', 'future_absolute_tangential_work_J',
        'future_signed_normal_work_J', 'future_absolute_normal_work_J')
CELL = ['bank', 'normal', 'case', 'history', 'profile_index', 'sign', 'duration',
        'strength', 'sensor', 'future', 'channel']
PHYSICAL_CELL = ['bank', 'normal', 'case', 'sign', 'duration', 'strength', 'future', 'channel']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write(path, value, exclusive=True):
    with Path(path).open('x' if exclusive else 'w') as stream:
        json.dump(clean_json(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def check(mapping):
    for path, digest in mapping.items():
        if sha(path) != digest:
            raise ValueError('Changed bound source/input: ' + str(path))


def finite(value, shape, name, nonnegative=False):
    value = np.asarray(value)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError('Invalid finite array shape: ' + name)
    if nonnegative and np.any(value < 0):
        raise ValueError('Negative values: ' + name)
    return value


def score_arrays(predictions, truth):
    """N,F,E,M,T,C predictions; F,E,T,C truth, with time pooled first."""
    pred, target = np.asarray(predictions, float), np.asarray(truth, float)
    if (pred.ndim != 6 or target.ndim != 4 or pred.shape[-1] != 4
            or (pred.shape[1], pred.shape[2], pred.shape[4], pred.shape[5]) != target.shape
            or not np.isfinite(pred).all() or not np.isfinite(target).all()):
        raise ValueError('Finite matching prediction and truth arrays required')
    error = pred - target[None, :, :, None]
    mse = np.mean(error * error, axis=-2)
    return dict(mse=mse, rmse=np.sqrt(mse), bias=np.mean(error, axis=-2))


def weight_metrics(weights):
    w = np.asarray(weights, float)
    if w.ndim != 3 or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError('Finite nonnegative N,E,P weights required')
    if not np.allclose(w.sum(axis=-1), 1., atol=1e-12, rtol=1e-12):
        raise ValueError('Particle weights must sum to one')
    logw = np.zeros_like(w)
    np.log(w, out=logw, where=w > 0)
    return dict(ess=1. / np.sum(w*w, axis=-1), entropy=-np.sum(w*logw, axis=-1),
                maximum_weight=np.max(w, axis=-1))


def expected_inventory(data, cfg):
    paths, identities = set(), {}
    signs = sorted(set(cfg['sign_assignment']))
    if len(cfg['sign_assignment']) != cfg['profiles_per_history']:
        raise ValueError('Profile/sign configuration mismatch')
    for bank, normal, case in itertools.product(cfg['banks'], cfg['normals'], cfg['cases']):
        model = f'bank{bank}_{normal}_{case}'
        paths.add(data / 'preparations' / (model + '_prior.npz'))
        for history in cfg['history_seeds']:
            paths.add(data / 'preparations' / f'{model}_h{history}.npz')
        for sign, duration, strength in itertools.product(signs, cfg['durations'], cfg['strengths_N']):
            arm = f's{sign:+d}_d{round(duration*1000)}_a{strength:g}.npz'
            ensemble = data / 'ensembles' / model / arm
            paths.add(ensemble)
            for history in cfg['history_seeds']:
                path = data / 'batches' / model / f'h{history}' / arm
                paths.add(path)
                identities[path] = dict(bank=bank, normal=normal, case=case, history=history,
                    sign=sign, duration=duration, strength=strength, ensemble_path=str(ensemble))
    return paths, identities


def batch_rows(arrays, meta, cfg):
    sensors, futures, methods = list(cfg['sensor_sigma']), cfg['future_amplitudes_N'], list(METHODS)
    if not cfg.get('mle_enabled', True):
        methods = [m for m in methods if not m.startswith('mle_')]
    for key, expected in [('sensors', sensors), ('futures', futures), ('methods', methods),
                          ('channel_order', list(CHANNELS)), ('dt', cfg['dt']), ('particles', cfg['particles'])]:
        if meta[key] != expected:
            raise ValueError('Batch configuration mismatch: ' + key)
    indices = np.flatnonzero(np.asarray(cfg['sign_assignment']) == meta['sign'])
    if not np.array_equal(arrays['profile_indices'], indices):
        raise ValueError('Profile/sign factorial identities mismatch')
    n, f, e, m, p = len(sensors), len(futures), len(indices), len(methods), cfg['particles']
    # Core returns canonical 2.5-ms output blocks even under finer integration.
    t, samples = round(cfg['future_duration']/.0025), round(meta['duration']/.025)
    truth = finite(arrays['truth_future'], (f,e,t,4), 'truth_future')
    prediction = finite(arrays['predictions'], (n,f,e,m,t,4), 'predictions')
    finite(arrays['truth_past'], (e,samples,3), 'truth_past')
    finite(arrays['observed'], (n,e,samples,3), 'observed')
    finite(arrays['noise_standard'], (e,samples,3), 'noise_standard')
    sigma = finite(arrays['effective_sigma'], (n,3), 'effective_sigma')
    if np.any(sigma <= 0):
        raise ValueError('Positive effective likelihood scales required')
    expected_sigma = np.asarray([np.hypot(cfg['sensor_sigma'][s], cfg['numerical_floor']) for s in sensors])
    if not np.array_equal(sigma, expected_sigma):
        raise ValueError('Likelihood scales differ from frozen configuration')
    prior_var = finite(arrays['prior_variance'], (f,t,4), 'prior_variance', True)
    post_var = finite(arrays['posterior_variance'], (n,f,e,t,4), 'posterior_variance', True)
    weights = finite(arrays['weights'], (n,e,p), 'weights', True)
    wm, score = weight_metrics(weights), score_arrays(prediction, truth)
    pv, cv = prior_var.mean(axis=1), post_var.mean(axis=-2)
    if cfg.get('mle_enabled', True):
        costs = finite(arrays['mle_cost'], (n,e,2), 'mle_cost', True)
        chosen = finite(arrays['mle_chosen'], (n,e), 'mle_chosen')
        finite(arrays['mle_initial'], (n,e,2,3), 'mle_initial')
        if not np.array_equal(chosen, np.argmin(costs, axis=-1)):
            raise ValueError('Stored MLE choice must retain minimum past objective and first ties')
    base = {key: meta[key] for key in ('bank','normal','case','history','sign','duration','strength','dt','particles')}
    rows, information, physical = [], [], []
    for fi, future in enumerate(futures):
        diag = meta['diagnostics'][fi]
        work = {key: finite(diag[key], (e,), key) for key in WORK}
        for ci, channel in enumerate(CHANNELS):
            physical.append(dict(**{k:v for k,v in base.items() if k != 'history'},
                ensemble_path=meta['ensemble_path'], ensemble_sha256=meta['ensemble_sha256'],
                future=future, channel=channel, prior_variance=float(pv[fi,ci])))
        for ni, sensor in enumerate(sensors):
            for ei, profile in enumerate(indices):
                extra = {key: float(values[ei]) for key, values in work.items()}
                common = dict(**base, sensor=sensor, future=future, profile_index=int(profile),
                    ess=float(wm['ess'][ni,ei]), entropy=float(wm['entropy'][ni,ei]),
                    maximum_weight=float(wm['maximum_weight'][ni,ei]), **extra)
                for ci, channel in enumerate(CHANNELS):
                    common_c = dict(**common, channel=channel, prior_variance=float(pv[fi,ci]),
                                    posterior_variance=float(cv[ni,fi,ei,ci]))
                    information.append(common_c)
                    for mi, method in enumerate(methods):
                        rows.append(dict(**common_c, method=method,
                            mse=float(score['mse'][ni,fi,ei,mi,ci]),
                            rmse=float(score['rmse'][ni,fi,ei,mi,ci]),
                            bias=float(score['bias'][ni,fi,ei,mi,ci])))
    return rows, information, physical


def deduplicate_physical(frame):
    keys = ['ensemble_path', 'future', 'channel']
    for _, group in frame.groupby(keys, sort=True):
        if any(group[col].nunique(dropna=False) != 1 for col in frame.columns):
            raise ValueError('Shared ensemble physical variance/metadata changed across histories')
    result = frame.drop_duplicates(keys).copy()
    if result.duplicated(PHYSICAL_CELL).any():
        raise ValueError('Duplicate physical factorial cells')
    return result


def pool(frame, groups, values):
    """Equal-weight complete truth contexts; RMS is sqrt of pooled MSE."""
    result = frame.groupby(groups, sort=True, dropna=False).agg(
        contexts=(values[0], 'size'), **{name:(name,'mean') for name in values}).reset_index()
    if 'mse' in values:
        result['pooled_rmse'] = np.sqrt(result['mse'])
        result = result.rename(columns={'rmse':'mean_episode_rmse'})
    return result


def crossed_summary(frame, numerator, denominator, cfg):
    """Ratio of balanced cell means; percentile resampling is descriptive."""
    cells = frame.groupby(['bank','history'], sort=True)[[numerator,denominator]].mean()
    banks, histories = sorted(frame.bank.unique()), sorted(frame.history.unique())
    expected = pd.MultiIndex.from_product([banks,histories], names=['bank','history'])
    cells = cells.reindex(expected)
    if cells.isna().any().any():
        raise ValueError('Incomplete crossed bank/history cells')
    count = frame.groupby(['bank','history']).size().reindex(expected)
    if count.nunique() != 1:
        raise ValueError('Unbalanced truth contexts in crossed comparison')
    x = cells[numerator].to_numpy().reshape(len(banks),len(histories))
    y = cells[denominator].to_numpy().reshape(x.shape)
    mx,my = float(x.mean()),float(y.mean())
    result = dict(numerator_mean=mx,denominator_mean=my,difference=mx-my,
        ratio=mx/my if my>0 else None,banks=len(banks),history_blocks=len(histories),
        crossed_cells=x.size,contexts=len(frame),bootstrap_draws=0,
        ratio_low=None,ratio_high=None,difference_low=None,difference_high=None,
        bootstrap_scope='descriptive crossed bank/history sensitivity; no significance decision')
    if min(x.shape)<2:
        result['interval_status']='unavailable: fewer than two banks or history blocks'
        return result
    spec=cfg['bootstrap'];draws=int(spec['draws']);rng=np.random.default_rng(spec['seed'])
    bi=rng.integers(len(banks),size=(draws,len(banks)))
    hi=rng.integers(len(histories),size=(draws,len(histories)))
    rx=x[bi[:,:,None],hi[:,None,:]].mean(axis=(1,2))
    ry=y[bi[:,:,None],hi[:,None,:]].mean(axis=(1,2))
    low,high=spec['interval']
    result.update(bootstrap_draws=draws,difference_low=float(np.quantile(rx-ry,low)),
                  difference_high=float(np.quantile(rx-ry,high)),
                  undefined_ratio_draws=int(np.count_nonzero(ry<=0)))
    if np.all(ry>0):
        q=np.quantile(rx/ry,[low,high]);result.update(ratio_low=float(q[0]),ratio_high=float(q[1]),interval_status='available; descriptive only')
    else:
        result['interval_status']='ratio unavailable: zero denominator; no bootstrap draws discarded'
    return result


def paired(frame, numerator, denominator, factor, value):
    keys = [c for c in CELL if c != factor]
    if 'method' in frame and factor!='method':
        keys += ['method']
    a=frame[frame[factor]==numerator][keys+[value]].rename(columns={value:'numerator'})
    b=frame[frame[factor]==denominator][keys+[value]].rename(columns={value:'denominator'})
    joined=a.merge(b,on=keys,how='outer',indicator=True,validate='one_to_one')
    if not (joined['_merge']=='both').all():
        raise ValueError('Incomplete paired contrast')
    return joined.drop(columns='_merge')


def compare_groups(frame, groups, cfg, label):
    rows=[]
    for key, group in frame.groupby(groups, sort=True, dropna=False):
        key=key if isinstance(key,tuple) else (key,)
        rows.append(dict(contrast=label,**dict(zip(groups,key)),**crossed_summary(group,'numerator','denominator',cfg)))
    return rows


def comparisons(metrics, information, physical, cfg):
    scopes=[['sensor','duration','future','channel'],
            ['sensor','duration','strength','future','channel'],
            ['normal','case','sensor','duration','strength','future','channel']]
    rows=[]
    method_pairs=[('posterior','prior')]
    if cfg.get('mle_enabled',True):
        method_pairs += [('mle_start1','posterior'),('mle_two_start','posterior'),('mle_two_start','mle_start1')]
    for a,b in method_pairs:
        joined=paired(metrics,a,b,'method','mse')
        for i,groups in enumerate(scopes):
            label='conditioning' if (a,b)==('posterior','prior') else 'estimator'
            rows.extend(dict(scope=f'{label}_{i}',**r) for r in compare_groups(joined,groups,cfg,f'{a}/{b} MSE'))
    factor_pairs=[('strength',a,b) for a,b in [(3.,0.),(12.,0.),(12.,3.)]
                  if a in cfg['strengths_N'] and b in cfg['strengths_N']]
    if .15 in cfg['durations'] and .30 in cfg['durations']:
        factor_pairs.append(('duration',.30,.15))
    for factor,a,b in factor_pairs:
        joined=paired(metrics,a,b,factor,'mse')
        held='duration' if factor=='strength' else 'strength'
        label=f'{a:g}/{b:g} '+('N' if factor=='strength' else 's')
        for groups in (['sensor',held,'future','method','channel'],
                       ['normal','case','sensor',held,'future','method','channel']):
            rows.extend(dict(scope='equal_duration_strength' if factor=='strength' else 'equal_strength_duration',**r)
                        for r in compare_groups(joined,groups,cfg,label+' MSE'))
    info=information.rename(columns={'posterior_variance':'numerator','prior_variance':'denominator'})
    info_rows=[]
    for i,groups in enumerate(scopes):
        info_rows.extend(dict(scope=f'information_{i}',**r) for r in compare_groups(info,groups,cfg,'posterior/prior variance'))
    physical_rows=[]
    for factor,numerator,denominator in factor_pairs:
        keys=[k for k in PHYSICAL_CELL if k!=factor]
        a=physical[physical[factor]==numerator][keys+['prior_variance']].rename(columns={'prior_variance':'numerator'})
        b=physical[physical[factor]==denominator][keys+['prior_variance']].rename(columns={'prior_variance':'denominator'})
        joined=a.merge(b,on=keys,how='outer',indicator=True,validate='one_to_one')
        if not (joined['_merge']=='both').all():raise ValueError('Incomplete physical arm pairing')
        held='duration' if factor=='strength' else 'strength'
        label=f'{numerator:g}/{denominator:g} '+('N' if factor=='strength' else 's')
        for groups in ([held,'future','channel'],['normal','case',held,'future','channel'],['bank',held,'future','channel']):
            for key,g in joined.groupby(groups,sort=True):
                key=key if isinstance(key,tuple) else (key,)
                x,y=float(g.numerator.mean()),float(g.denominator.mean())
                physical_rows.append(dict(contrast=label+' prior variance',**dict(zip(groups,key)),
                    numerator_mean=x,denominator_mean=y,difference=x-y,ratio=x/y if y>0 else None,
                    ensemble_contexts=len(g),bootstrap_scope='none; no history replication of shared ensemble'))
    return pd.DataFrame(rows),pd.DataFrame(info_rows),pd.DataFrame(physical_rows)


def truth_work(metrics):
    """One physical work record per truth/probe/future; no estimator/noise repeats."""
    keys=['bank','normal','case','history','profile_index','sign','duration','strength','future']
    for _,group in metrics.groupby(keys,sort=False):
        if any(group[key].nunique(dropna=False)!=1 for key in WORK):
            raise ValueError('Physical work changed across methods/sensors/channels')
    return metrics[keys+list(WORK)].drop_duplicates(keys).copy()


def figures(metrics, physical, out, cfg):
    plt.rcParams.update({'font.size':9.5,'pdf.fonttype':42,'ps.fonttype':42})
    styles=dict(zip(METHODS,[('#777777','s'),('#0072B2','o'),('#D55E00','^'),('#009E73','v'),('#202020','x')]))
    compositions=list(itertools.product(cfg['normals'],cfg['cases']))
    sensors=list(cfg['sensor_sigma']);files=[]
    values=metrics[metrics.channel=='position_m']
    for duration,future in itertools.product(cfg['durations'],cfg['future_amplitudes_N']):
        height=2.5*len(compositions)+1.6
        fig,axes=plt.subplots(len(compositions),len(sensors),figsize=(7.,height),squeeze=False)
        for ri,(normal,case) in enumerate(compositions):
            for ni,sensor in enumerate(sensors):
                ax=axes[ri,ni]
                d=values[(values.normal==normal)&(values.case==case)&(values.sensor==sensor)&(values.duration==duration)&(values.future==future)]
                means=d.groupby(['strength','method']).mse.mean()
                for method in METHODS:
                    if method not in d.method.unique():continue
                    y=[means.loc[(x,method)]*1e6 for x in cfg['strengths_N']]
                    color,marker=styles[method];ax.plot(cfg['strengths_N'],y,color=color,marker=marker,label=method,lw=1.1,ms=4)
                ax.set_title(f'{normal}/{case}\n{sensor}',fontsize=9.5)
                ax.set_yscale('symlog',linthresh=1e-8);ax.grid(alpha=.2)
                ax.yaxis.get_major_locator().set_params(numticks=7)
                ax.set_xticks(cfg['strengths_N'])
                if ni==0:ax.set_ylabel('Positions-MSE (mm²)')
                if ri==len(compositions)-1:ax.set_xlabel('Anregungsamplitude (N)')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=9,frameon=False)
        fig.suptitle(f'{duration*1000:g} ms Präparation; {future:g} N Zukunft',y=1-.58/height,fontsize=10)
        fig.tight_layout(rect=(0,.32/height,1,1-.88/height))
        fig.text(.5,.08/height,'Linien verbinden getestete Amplituden. Gepoolte MSE; symlog-Skala.',ha='center',fontsize=9)
        name=f'position_mse_d{round(duration*1000)}_future{future:g}'
        for suffix in ('.png','.pdf'):
            dest=out/(name+suffix);fig.savefig(dest,dpi=220);files.append(dest.name)
        plt.close(fig)
    # A separate plot has no sensor/history duplicates and no conditioning.
    height=2.3*len(compositions)+1.2
    fig,axes=plt.subplots(len(compositions),1,figsize=(7.,height),squeeze=False)
    colors=['#0072B2','#D55E00','#009E73','#CC79A7']
    for ri,(normal,case) in enumerate(compositions):
        ax=axes[ri,0]
        d=physical[(physical.normal==normal)&(physical.case==case)&(physical.channel=='position_m')]
        for i,(duration,future) in enumerate(itertools.product(cfg['durations'],cfg['future_amplitudes_N'])):
            means=d[(d.duration==duration)&(d.future==future)].groupby('strength').prior_variance.mean()
            ax.plot(means.index,means.to_numpy()*1e6,marker='o' if i%2==0 else '^',color=colors[i%len(colors)],label=f'{duration*1000:g} ms / Zukunft {future:g} N')
        ax.set_title(f'{normal}/{case}',fontsize=9.5);ax.set_ylabel('Prior-Varianz (mm²)')
        ax.set_yscale('symlog',linthresh=1e-8);ax.set_xticks(cfg['strengths_N']);ax.grid(alpha=.2)
        ax.yaxis.get_major_locator().set_params(numticks=7)
    axes[-1,0].set_xlabel('Anregungsamplitude (N)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='upper center',frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.32/height,1,1-.55/height))
    fig.text(.5,.08/height,'Gleiches Ensemble ohne Assimilation; keine Historien- oder Rauschreplikation.',ha='center',fontsize=9)
    for suffix in ('.png','.pdf'):
        dest=out/('physical_prior_variance'+suffix);fig.savefig(dest,dpi=220);files.append(dest.name)
    plt.close(fig)
    # Preparation work is independent of the two later branch amplitudes too.
    work=truth_work(metrics)
    prepkeys=['bank','normal','case','history','profile_index','sign','duration','strength']
    for _,g in work.groupby(prepkeys,sort=False):
        if any(g[key].nunique()!=1 for key in WORK if key.startswith('prep_')):
            raise ValueError('Preparation work depends on future branch')
    work=work.drop_duplicates(prepkeys)
    height=2.5*len(compositions)+1.3
    fig,axes=plt.subplots(len(compositions),2,figsize=(7.,height),squeeze=False)
    for ri,(normal,case) in enumerate(compositions):
        d=work[(work.normal==normal)&(work.case==case)]
        for ci,key in enumerate(('prep_signed_tangential_work_J','prep_absolute_tangential_work_J')):
            ax=axes[ri,ci]
            for i,duration in enumerate(cfg['durations']):
                g=d[d.duration==duration].groupby('strength')[key]
                means=g.mean();low=g.min();high=g.max()
                ax.plot(means.index,means.to_numpy()*1e3,color=colors[i%len(colors)],marker='o' if i==0 else '^',label=f'{duration*1000:g} ms')
                ax.fill_between(means.index,low.to_numpy()*1e3,high.to_numpy()*1e3,color=colors[i%len(colors)],alpha=.12)
            ax.set_title(f'{normal}/{case}',fontsize=9.5);ax.grid(alpha=.2)
            ax.set_ylabel(('Signierte' if ci==0 else 'Absolute')+' Arbeit (mJ)')
            ax.set_xticks(cfg['strengths_N'])
            if ri==len(compositions)-1:ax.set_xlabel('Anregungsamplitude (N)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='upper center',frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.54/height,1,1-.38/height))
    fig.text(.5,.08/height,'Mittel und Min–Max über Wahrheitskontexte; gekoppelte Probe, ohne 6-s-Fixture.\n0 N: null Antriebsarbeit, Bewegung durch gespeicherte Kräfte möglich.',ha='center',fontsize=9)
    for suffix in ('.png','.pdf'):
        dest=out/('truth_preparation_work'+suffix);fig.savefig(dest,dpi=220);files.append(dest.name)
    plt.close(fig)
    return files


def report(out, cfg, metrics, info, physical, contrasts, summary, plots):
    lines=['# Kontrollierte Matched-Model-Studie: deskriptive Auswertung','',
        ('Der Analyzer war vor der Datenerzeugung in deren Quellensperre enthalten.' if summary['analyzer_frozen_before_generation'] else 'Ausschließlich technische Entwicklungsauswertung: Die Analyzerquelle wurde erst nach der Datenerzeugung separat gesperrt. Diese Daten sind keine Hauptstudienresultate.'),'',
        f"Vollständige konfigurierte Population: {summary['truth_preparation_contexts']} Präparationsfälle, {summary['forecast_contexts']} Sensor/Zukunftsfälle und {len(metrics)} Methoden/Kanal-Zeilen. {len(cfg['banks'])} Modellbanken und {len(cfg['history_seeds'])} Historienblöcke sind gekreuzt; dieselbe Historie und dasselbe Ensemble werden mehrfach verwendet. Dies sind keine unabhängigen Zeilenstichproben.",'',
        'Wahrheit und Schätzung verwenden dieselben eingefrorenen Gesetze. Bekannte Normalvorbereitung wird getragen; Kopplungsbeginn bei 0,02 m Kompression. Gemessen werden Prognosefehler und Zukunftsstreuung unter festen offenen Befehlen. Keine neue Lernmethode, kein nativer PhysX-Test und kein realer Roboter sind Gegenstand dieser Auswertung.','',
        'Die Primärgröße ist der Mittelwert der quadratischen Fehler über Zeit und ausgewogene Wahrheitskontexte. Angegebene gepoolte RMSE ist dessen Quadratwurzel; der Mittelwert einzelner RMSE steht getrennt in den CSVs.','',
        '| Sensor | Dauer (ms) | Zukunft (N) | Posterior/Prior-MSE | 99%-Bootstrapbereich |',
        '|---|---:|---:|---:|---|']
    c=contrasts[(contrasts.scope=='conditioning_0')&(contrasts.channel=='position_m')]
    def number(x):return 'nicht definiert' if x is None or pd.isna(x) else f'{x:.4g}'
    for row in c.itertuples():
        interval=f'[{number(row.ratio_low)}, {number(row.ratio_high)}]' if pd.notna(row.ratio_low) else 'nicht verfügbar'
        lines.append(f'| {row.sensor} | {row.duration*1000:g} | {row.future:g} | {number(row.ratio)} | {interval} |')
    lines += ['',f"Die {cfg['bootstrap']['draws']:,} gepaarten Bootstrapziehungen (Seed {cfg['bootstrap']['seed']}) resampeln Bank- und Historienachsen separat. Bei nur {len(cfg['banks'])} Banken × {len(cfg['history_seeds'])} Blöcken sind die 99%-Bereiche eine deskriptive Sensitivität, keine formale Signifikanz- oder Abdeckungszusage. Bei weniger als zwei Einheiten je Achse werden keine Bereiche angegeben. Nullnenner werden nicht durch Epsilon ersetzt und keine ungültigen Ziehungen verworfen.",'',
        'Die Amplitudenvergleiche 3/0, 12/0 und 12/3 N verwenden jeweils dieselbe Dauer. Der 300/150-ms-Vergleich hält die Amplitude konstant. MLE1/PF, MLE2/PF und MLE2/MLE1 verwenden dieselben Sensor-/Proben-/Zukunftskontexte. Alle im jeweiligen Lauf verfügbaren Kontraste und vier Kanäle bleiben in `mse_contrasts.csv`; es wird keine günstige Methode oder Teilgruppe zur Erfolgsdefinition ausgewählt.','',
        f"Physikalische Prior-Streuung wird über {summary['unique_ensemble_future_contexts']} eindeutige Ensemble/Zukunftskontexte berechnet und nicht durch die Historienblöcke vervielfacht. `physical_contrasts.csv` enthält dieselben verfügbaren Amplituden- und Dauerverhältnisse ohne Historien-Bootstrap. `information_contrasts.csv` vergleicht dagegen posterior- und priorgewichtete Varianz mit den tatsächlichen Beobachtungskontexten.",'',
        'Eine kleinere Varianz belegt für sich keine höhere Genauigkeit oder Kalibrierung. Einzelne Messungen dürfen die posteriorgewichtete Varianz erhöhen. Ein kleinerer Prior-Zukunftsabstand betrifft das angegebene Ausgabeprogramm; er beweist keine Gleichheit aller inneren Zustände oder ausschließliches Löschen von Reibungsgedächtnis.','',
        f"ESS-Bereich: {info.ess.min():.4g} bis {info.ess.max():.4g} bei P={cfg['particles']}. Maximalgewicht bis {info.maximum_weight.max():.4g}. Maximaler privilegierter Positions-MSE: {metrics[(metrics.method=='privileged_history')&(metrics.channel=='position_m')].mse.max():.4g} m². Ein exakter privilegierter Replay ist eine Implementierungskontrolle, kein Nachweis der Zeitraster-Konvergenz.",'',
        'Die gespeicherten Arbeitsgrößen erfassen die gekoppelte Präparationsphase und die Zukunft separat; die gemeinsame sechssekündige anfängliche Materialfixture ist darin nicht als variable Probenarbeit enthalten. Ein offenes 0-N-Programm ist keine Positionsklemme. Normalantriebsarbeit bei konstantem d kann null sein, obwohl Normalkräfte über die Geometrie wirken.','',
        'Endlicher Prior-Support und begrenzte MLE-Optimierung bleiben auch ohne Sensorrauschen mögliche Fehlerursachen. Die MLE-Box umfasst nicht zwingend nur physikalisch erreichte Zustände. Die getrennten P- und h/h2/h4-Läufe müssen einschließlich Anfangszustandsdrift und Effektkontrasten ausgewertet werden; dieser einzelne Lauf ersetzt diese Kontrollen nicht. Die vier Kompositionen bilden den vorab begrenzten Untersuchungsbereich.','',
        'Alle Aufschlüsselungen stehen in `metrics_by_context.csv`, `metrics_by_bank.csv`, `metrics_by_history.csv`, `information_by_context.csv` und `physical_by_context.csv`. Rohwerte: `episode_metrics.csv`, `information_metrics.csv`, `physical_metrics.csv`. Quell-/Inputbindung, Vollständigkeit und numerische Diagnosewerte: `summary.json`, `analysis_lock.json`, `manifest.json`.','',
        'Abbildungen (alle konfigurierten Dauern und Zukunftsprogramme):','']
    for name in plots:
        if name.endswith('.png'):lines.append(f'![{name}]({name})')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run(data, out):
    data,out=Path(data).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Use a fresh analysis output directory')
    started=time.perf_counter();out.mkdir(parents=True)
    process=dict(status='running',started_unix=time.time(),data_directory=str(data),source_sha256=sha(__file__),command=sys.argv)
    write(out/'process.json',process)
    try:
        controls={str(data/name):sha(data/name) for name in ('lock.json','process.json','manifest.json')}
        lock,upstream,manifest=read(data/'lock.json'),read(data/'process.json'),read(data/'manifest.json')
        digest=controls[str(data/'lock.json')]
        if upstream.get('status')!='complete' or upstream.get('lock_sha256')!=digest or manifest.get('lock_sha256')!=digest:
            raise ValueError('Complete source-bound run/process/manifest required')
        cfg=lock['config']
        analyzer_source={str(Path(__file__).resolve()):process['source_sha256']}
        prospective=lock['source_sha256'].get(str(Path(__file__).resolve()))==process['source_sha256']
        if not prospective and cfg.get('kind')!='development':
            raise ValueError('Analyzer must be frozen in the source lock before the study run')
        check(controls);check(lock['source_sha256']);check(analyzer_source);check(manifest['outputs_sha256'])
        expected,identities=expected_inventory(data,cfg)
        actual={Path(p).resolve() for p in manifest['outputs_sha256'] if Path(p).suffix=='.npz'}
        disk={p.resolve() for p in data.rglob('*.npz')}
        if actual!=expected or disk!=expected or upstream.get('batches')!=len(identities):
            raise ValueError('Exact configured preparation/ensemble/batch inventory required')
        write(out/'analysis_lock.json',dict(source_sha256=lock['source_sha256'],analyzer_source_sha256=analyzer_source,
            analyzer_frozen_before_generation=prospective,run_lock_sha256=digest,
            control_sha256=controls,input_sha256=manifest['outputs_sha256'],config=cfg,
            scope=('Prospective source-bound descriptive analysis' if prospective else 'Development only: analyzer separately bound after generation')+'; no inference or population claim from row count'))
        process['analysis_lock_sha256']=sha(out/'analysis_lock.json');write(out/'process.json',process,False)
        rows,info,physical=[],[],[];diag=[]
        for path,identity in sorted(identities.items()):
            with np.load(path,allow_pickle=False) as stored:
                meta=json.loads(str(stored['metadata_json'].item()))
                for key,value in identity.items():
                    if meta.get(key)!=value:raise ValueError('Factor/path metadata mismatch: '+str(path)+' '+key)
                if meta['ensemble_sha256']!=manifest['outputs_sha256'].get(meta['ensemble_path']):
                    raise ValueError('Shared ensemble must be hash-bound by the same run')
                r,i,p=batch_rows(stored,meta,cfg)
            rows.extend(r);info.extend(i);physical.extend(p)
            diag.extend(meta['diagnostics'])
        metrics=pd.DataFrame(rows);information=pd.DataFrame(info);physical=deduplicate_physical(pd.DataFrame(physical))
        if metrics.duplicated(CELL+['method']).any() or information.duplicated(CELL).any():
            raise ValueError('Repeated truth context identities')
        contrasts,info_contrasts,physical_contrasts=comparisons(metrics,information,physical,cfg)
        values=['mse','rmse','bias',*WORK]
        work=truth_work(metrics)
        tables={'episode_metrics':metrics,'information_metrics':information,'physical_metrics':physical,'truth_work_metrics':work,
            'mse_contrasts':contrasts,'information_contrasts':info_contrasts,'physical_contrasts':physical_contrasts,
            'metrics_pooled':pool(metrics,['method','channel'],values),
            'metrics_by_context':pool(metrics,['normal','case','sensor','duration','strength','future','method','channel'],values),
            'metrics_by_bank':pool(metrics,['bank','sensor','duration','strength','future','method','channel'],values),
            'metrics_by_history':pool(metrics,['history','sensor','duration','strength','future','method','channel'],values),
            'information_by_context':pool(information,['normal','case','sensor','duration','strength','future','channel'],['prior_variance','posterior_variance','ess','entropy','maximum_weight']),
            'physical_by_context':pool(physical,['normal','case','duration','strength','future','channel'],['prior_variance'])}
        for name,frame in tables.items():frame.to_csv(out/(name+'.csv'),index=False)
        summary=dict(config=cfg,batches=len(identities),metric_rows=len(metrics),analyzer_frozen_before_generation=prospective,
            truth_preparation_contexts=len(information.drop_duplicates(['bank','normal','case','history','profile_index','sign','duration','strength'])),
            forecast_contexts=len(information.drop_duplicates([c for c in CELL if c!='channel'])),
            unique_ensemble_future_contexts=len(physical.drop_duplicates(['ensemble_path','future'])),
            numerics={'max_solver_residual_N':max(float(d['max_solver_residual_N']) for d in diag)},
            privileged_mse_max=float(metrics[metrics.method=='privileged_history'].mse.max()),
            posterior_variance_larger_than_prior_rows=int((information.posterior_variance>information.prior_variance).sum()),
            formal_significance_decision=None,limitations=['Three fitted banks and four history blocks in the main design; crossed bootstrap descriptive only.',
                'Finite-support Bayesian approximation; no guaranteed calibration.',
                'Separate refinement/support diagnostics required for continuous-physics or resolution claims.'])
        if summary['numerics']['max_solver_residual_N']>cfg['maximum_solver_residual_N']:
            raise ValueError('Declared solver residual bound exceeded')
        write(out/'summary.json',summary)
        plots=figures(metrics,physical,out,cfg);report(out,cfg,metrics,information,physical,contrasts,summary,plots)
        check(controls);check(lock['source_sha256']);check(analyzer_source);check(manifest['outputs_sha256'])
        outputs={str(p):sha(p) for p in out.iterdir() if p.is_file() and p.name!='process.json'}
        write(out/'manifest.json',dict(analysis_lock_sha256=process['analysis_lock_sha256'],source_sha256=process['source_sha256'],outputs_sha256=outputs))
        write(out/'process.json',dict(**{k:v for k,v in process.items() if k!='status'},status='complete',seconds=time.perf_counter()-started,metric_rows=len(metrics)),False)
        print(json.dumps(clean_json(summary),indent=2))
    except BaseException as exc:
        outputs={str(p):sha(p) for p in out.iterdir() if p.is_file() and p.name!='process.json'}
        write(out/'process.json',dict(**{k:v for k,v in process.items() if k!='status'},status='failed',error=repr(exc),seconds=time.perf_counter()-started,partial_outputs_sha256=outputs),False)
        raise
    finally:
        plt.close('all')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.data,args.out)

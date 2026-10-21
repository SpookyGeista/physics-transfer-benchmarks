"""Presentation-only vector/PNG figures from unchanged audited Main3 summaries."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/decision-test-publication-mpl')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
import argparse
import hashlib
import json
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

STATISTICS = ROOT / 'results/statistics_decision_test'
METHODS = ['fixed_release', 'fixed_hold100', 'fixed_continue100', 'uniform',
           'uniform_expected', 'bayes_risk', 'information_gain', 'physical_only']
BUDGETS = ['B0', 'B1', 'B2', 'B3']
LABELS = ['Fixed release', 'Fixed hold', 'Fixed continue', 'Uniform draw',
          'Uniform expected loss', 'Bayes risk', 'Information gain', 'Physical only']
COLORS = ['#777777', '#E69F00', '#0072B2', '#56B4E9', '#CC79A7', '#D55E00', '#009E73', '#202020']
MARKERS = ['D', 's', '^', 'v', 'x', 'o', 'P', 'h']
LINES = ['-', '--', '-.', ':', (0, (4, 1, 1, 1)), '-', (0, (5, 2)), (0, (2, 1, 2, 3))]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify(mapping):
    for path, digest in mapping.items():
        if sha(path) != digest:
            raise ValueError('Changed presentation source/input: ' + str(path))


def rc():
    # Exact7-inch width: these are final-size fonts, not pre-scaling values.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.labelsize': 10, 'axes.titlesize': 10, 'xtick.labelsize': 10,
        'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.titlesize': 10,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .7,
        'xtick.major.width': .7, 'ytick.major.width': .7,
        'savefig.dpi': 300, 'figure.dpi': 120})


def save(fig, out, name, metadata):
    # No bbox_inches='tight': retain exactly7-inch manuscript width and font scale.
    fig.savefig(out / (name + '.pdf'), metadata={'Title': metadata,
        'Author': 'Physics-transfer simulation study', 'Subject': 'Descriptive fixed-budget results; no confidence intervals'})
    fig.savefig(out / (name + '.png'), dpi=300)
    plt.close(fig)


def work_risk(pooled, out):
    fig = plt.figure(figsize=(7., 5.))
    axis = fig.add_axes([.12, .33, .85, .57])
    handles = []
    for method, label, color, marker, line in zip(METHODS, LABELS, COLORS, MARKERS, LINES):
        values = pooled[pooled.method == method].set_index('budget').loc[BUDGETS]
        handle, = axis.plot(values.extra_probe_work_J.to_numpy() * 1000,
            values.position_mse_m2.to_numpy() * 1e9, color=color, linestyle=line,
            linewidth=1.2, marker=marker, markersize=6.2, markeredgewidth=1.1,
            markerfacecolor='none', label=label, zorder=3)
        handles.append(handle)
    axis.set_xlabel('Mean realized additional probe work (mJ)', labelpad=5)
    axis.set_ylabel('Future position MSE ($10^{-9}$ m$^2$)', labelpad=5)
    axis.set_xlim(-.1, 4.12)
    axis.set_ylim(4.8, 17.85)
    axis.set_yticks([5, 7.5, 10, 12.5, 15, 17.5])
    axis.grid(axis='both', color='.88', linewidth=.5, zorder=0)
    axis.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .965, 'Discrete budgets B0–B3; lines join budget points only', ha='center', va='top', fontsize=10)
    fig.legend(handles, LABELS, loc='upper center', bbox_to_anchor=(.51, .245),
               ncol=2, frameon=False, columnspacing=3., handlelength=3.,
               handletextpad=.7, borderaxespad=0., labelspacing=.5)
    fig.text(.5, .025, 'All 8 methods and 4 budgets shown. Coincident budget points overlap.',
             ha='center', va='bottom', fontsize=10)
    save(fig, out, 'work_and_forecast_loss', 'Main3 work and forecasting loss at four discrete budgets')


def contrasts_heatmap(contrasts, out):
    values = contrasts[contrasts.method == 'bayes_risk']
    column = 'position_mse_m2_difference_vs_fixed_continue100'
    maximum = float(abs(values[column]).max()) * 1e9
    scale = max(maximum, np.finfo(float).tiny)
    normalizer = TwoSlopeNorm(vmin=-scale, vcenter=0., vmax=scale)
    fig, axes = plt.subplots(2, 2, figsize=(7., 6.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.805, top=.9, bottom=.11, wspace=.17, hspace=.22)
    for budget, axis in zip(BUDGETS, axes.flat):
        matrix = values[values.budget == budget].pivot(index='model_seed', columns='preparation_seed', values=column)
        matrix = matrix.loc[range(1, 9), range(55201, 55209)]
        # Vector rectangles (not an embedded raster image) in the PDF.
        mesh = axis.pcolormesh(np.arange(9), np.arange(9), matrix.to_numpy() * 1e9,
            cmap='RdBu_r', norm=normalizer, shading='flat', edgecolors='none', rasterized=False)
        axis.set_title(budget, pad=5)
        axis.set_aspect('equal')
        axis.set_xticks(np.arange(8) + .5, labels=range(1, 9))
        axis.set_yticks(np.arange(8) + .5, labels=range(1, 9))
        axis.set_ylim(8, 0)  # idempotent under shared axes: bank1 remains at the top
        axis.tick_params(length=0, pad=4)
    for axis in axes[-1]:
        axis.set_xlabel('History block', labelpad=4)
    for axis in axes[:, 0]:
        axis.set_ylabel('Fitted bank', labelpad=4)
    color_axis = fig.add_axes([.85, .19, .025, .64])
    colorbar = fig.colorbar(mesh, cax=color_axis)
    colorbar.solids.set_rasterized(False)
    # Overlap adjacent vector swatches to avoid renderer-only white seams.
    colorbar.solids.set_edgecolor('face')
    colorbar.ax.tick_params(labelsize=10)
    colorbar.set_label('MSE difference ($10^{-9}$ m$^2$); positive is worse', fontsize=10, labelpad=7)
    fig.text(.5, .965, 'Bayes risk − fixed continue: all budgets, banks and history blocks',
             ha='center', va='top', fontsize=10)
    fig.text(.5, .025, 'History blocks 1–8 correspond to seeds 55201–55208. Descriptive means; no CIs.',
             ha='center', va='bottom', fontsize=10)
    save(fig, out, 'bank_history_contrasts', 'Main3 Bayes-risk versus fixed-continuation contrasts')


def pdf_checks(out, names):
    evidence = {}
    for name in names:
        pdf = out / (name + '.pdf')
        evidence[name] = {}
        for tool in ('pdfinfo', 'pdffonts', 'pdfimages'):
            system_tool = Path('/usr/bin') / tool
            executable = str(system_tool) if system_tool.is_file() else shutil.which(tool)
            if executable is None:
                raise RuntimeError('Required publication QA tool missing: ' + tool)
            command = [executable, '-list', str(pdf)] if tool == 'pdfimages' else [executable, str(pdf)]
            output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
            path = out / (name + '_' + tool + '.txt')
            path.write_text(output)
            evidence[name][tool] = output
        if any(line.strip() and line.split()[0].isdigit() for line in evidence[name]['pdfimages'].splitlines()[2:]):
            raise ValueError('PDF contains a raster image instead of wholly vector marks')
        if '504 x ' not in evidence[name]['pdfinfo']:
            raise ValueError('PDF manuscript width must be exactly504points =7inches')
        system_tool = Path('/usr/bin/pdftoppm')
        executable = str(system_tool) if system_tool.is_file() else shutil.which('pdftoppm')
        if executable is None:
            raise RuntimeError('pdftoppm required for rendered PDF inspection')
        subprocess.run([executable, '-png', '-r', '150', '-singlefile', str(pdf),
                        str(out / (name + '_pdf_preview'))], check=True, capture_output=True)
    return evidence


def run(out):
    out = Path(out).resolve()
    allowed = ROOT / 'reports/figures'
    if out.exists() or not out.is_relative_to(allowed) or out == allowed:
        raise ValueError('Use a fresh named output directory under reports/figures')
    manifest = read(STATISTICS / 'artifact_manifest.json')
    verify(manifest['outputs_sha256'])
    inputs = [STATISTICS / 'pooled.csv', STATISTICS / 'contrasts_by_bank_history.csv',
              STATISTICS / 'analysis_lock.json', STATISTICS / 'artifact_manifest.json']
    hashes = {str(p): sha(p) for p in inputs}
    pooled, contrasts = pd.read_csv(inputs[0]), pd.read_csv(inputs[1])
    if len(pooled) != 32 or set(map(tuple, pooled[['budget', 'method']].to_numpy())) != set((b, m) for b in BUDGETS for m in METHODS):
        raise ValueError('All32 unchanged pooled budget/method rows are required')
    selected = contrasts[contrasts.method == 'bayes_risk']
    expected = {(b, s, h) for b in BUDGETS for s in range(1, 9) for h in range(55201, 55209)}
    if len(selected) != 256 or set(map(tuple, selected[['budget', 'model_seed', 'preparation_seed']].to_numpy())) != expected:
        raise ValueError('All256 pre-existing contrast cells are required')
    if not np.isfinite(pooled.select_dtypes(include=np.number)).all().all() or not np.isfinite(selected.select_dtypes(include=np.number)).all().all():
        raise ValueError('Nonfinite publication input')
    out.mkdir(parents=True)
    source = {str(Path(__file__).resolve()): sha(__file__)}
    write_json(out / 'figure_source_lock.json', dict(role='Presentation only; unchanged audited tables',
        source_sha256=source, input_sha256=hashes, created_unix=time.time(), command=sys.argv,
        operations='Existing rows only; unit conversion J→mJ and m²→10^-9m²; table reshaping, no statistics recomputation',
        manuscript_width_inches=7., minimum_font_points=10., png_dpi=300,
        matplotlib_version=matplotlib.__version__, numpy_version=np.__version__, pandas_version=pd.__version__))
    rc(); work_risk(pooled, out); contrasts_heatmap(contrasts, out)
    qa = pdf_checks(out, ('work_and_forecast_loss', 'bank_history_contrasts'))
    write_json(out / 'pdf_quality_checks.json', dict(vector_only=True, manuscript_width_inches=7.,
        minimum_font_points=10., pdf_render_dpi=150, checks=qa,
        human_visual_inspection='Separate visual inspection of rendered PDF previews required after generation'))
    captions = '''# Publication figure captions and sizing

**Figure5 — Forecast loss and realized additional preparation work under discrete budgets.**
All eight policies are shown at the four previously fixed budgets B0–B3. Points use the unchanged, audited pooled position MSE and mean realized additional probe work. Lines join successive budget points only; they do not imply interpolation, a continuous trade-off curve or a Pareto frontier. Coincident points overlap. Uniform expected loss averages action losses, not prediction trajectories. Budget eligibility is based on model-expected work; the horizontal axis reports realized work. Descriptive crossed eight-bank/eight-history-block summaries, without confidence intervals.

**Secondary figure — Bank/history variation in the prespecified contrast.**
Each cell is the unchanged Bayes-risk minus fixed-continuation position MSE for one fitted bank and one history block, averaged over the already specified compositions and contexts. All four budgets and all64 bank/history cells per budget are retained. Positive values indicate larger Bayes-risk loss. The color scale is shared across panels. History blocks1–8 denote seeds55201–55208. No new subgroup selection or statistical test is performed.

Both PDFs are exactly7inches wide. Base typography is at least10points at that width, retaining at least9points when reduced to6.3inches. Mathematical superscripts follow normal typesetting scale. The PDFs contain vector paths and embedded TrueType fonts, with no raster image objects. PNG exports are300dpi. The `*_pdf_preview.png` files are150dpi renders of the PDFs for visual QA, not manuscript replacements. Source/input hashes, renderer/font/image checks and output hashes accompany these files.
'''
    (out / 'CAPTIONS.md').write_text(captions)
    verify(hashes); verify(source); verify(manifest['outputs_sha256'])
    files = [p for p in out.iterdir() if p.is_file()]
    write_json(out / 'artifact_manifest.json', dict(figure_source_lock_sha256=sha(out / 'figure_source_lock.json'),
        outputs_sha256={str(p): sha(p) for p in files}))
    print('PUBLICATION FIGURES COMPLETE', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/figures/decision_test')
    run(parser.parse_args().out)

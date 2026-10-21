#!/usr/bin/env python3
"""Post-lock presentation export; consumes and verifies existing Main2 statistics.

No fitting, target reading, statistical rule change, or case selection is performed.
The original manifest-bound German figures are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper_study/results/statistics_main2"
DEST = ROOT / "paper_study/reports/figures/main2_final"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    names = ["main2_past_vs_future_en.png", "main2_past_vs_future_en.pdf", "provenance.json", "caption.md"]
    if any((destination / name).exists() for name in names):
        raise FileExistsError("Presentation outputs already exist; use a new destination")
    manifest_path = source / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = manifest["outputs_sha256"]
    for path, expected in hashes.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Changed original analysis artifact: {path}")
    input_path = source / "nested_mle.csv"
    frame = pd.read_csv(input_path)
    key = ["bank", "preparation_seed", "normal_world", "friction_world", "probe", "episode"]
    assert len(frame) == 10368 and not frame.duplicated(key).any()
    assert set(frame.bank) == {1, 2, 3}
    probes = ["hold_150", "gentle_150", "strong_300"]
    assert set(frame.probe) == set(probes)
    x = frame.past_improvement.to_numpy()
    y = frame.future_mse_change_position_m.to_numpy() * 1e6
    assert np.isfinite(x).all() and np.isfinite(y).all() and (x >= 0).all()
    assert (frame.groupby(["probe", "bank"]).size() == 1152).all()
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white",
    })
    colors = ["#0072B2", "#D55E00", "#009E73"]
    titles = ["Hold · 150 ms", "Gentle probe · 150 ms", "Strong probe · 300 ms"]
    fig, axes = plt.subplots(1, 3, figsize=(7, 3.35), sharex=True, sharey=True)
    fig.subplots_adjust(left=.152, right=.983, bottom=.24, top=.83, wspace=.13)
    plotted = 0
    for index, (ax, probe, title) in enumerate(zip(axes, probes, titles)):
        for bank, color in zip([1, 2, 3], colors):
            cell = frame[(frame.probe == probe) & (frame.bank == bank)]
            ax.scatter(cell.past_improvement, cell.future_mse_change_position_m * 1e6,
                       s=5, c=color, alpha=.43, linewidths=0, rasterized=False,
                       clip_on=False, zorder=3)
            plotted += len(cell)
        ax.set_xscale("symlog", linthresh=1e-8, linscale=.5)
        ax.set_yscale("symlog", linthresh=.001, linscale=.5)
        ax.set_xlim(0, 1000)
        ax.set_ylim(-1000, 1000)
        ax.xaxis.set_major_locator(FixedLocator([0, 1e-6, 1e-3, 1, 1000]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0", "$10^{-6}$", "$10^{-3}$", "1", "$10^3$"]))
        ax.yaxis.set_major_locator(FixedLocator([-100, -1, -.001, 0, .001, 1, 100]))
        ax.yaxis.set_major_formatter(FixedFormatter(["−100", "−1", "−0.001", "0", "0.001", "1", "100"]))
        ax.axhline(0, color=".35", lw=.6, zorder=1)
        ax.grid(axis="y", which="major", lw=.35, alpha=.23)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title(f"({chr(97 + index)}) {title}", pad=7)
    assert plotted == len(frame)
    axes[0].set_ylabel("Future position MSE change (mm²)\nselected two-start − start 1", labelpad=6)
    fig.supxlabel("Past objective improvement: $J_1-J_{\\mathrm{selected}}$", y=.11, fontsize=9)
    fig.text(.55, .035, "All 10,368 contexts; symmetric-log axes with linear regions near zero", ha="center", fontsize=7.5)
    fig.legend([Line2D([], [], marker="o", linestyle="", color=c, markersize=4) for c in colors],
               ["Calibration bank 1", "Calibration bank 2", "Calibration bank 3"],
               ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.55, 1.0))
    fig.savefig(destination / names[0], dpi=300)
    fig.savefig(destination / names[1])
    plt.close(fig)
    caption = """# Main2: past likelihood and future prediction

All 10,368 paired contexts are retained (three fitted banks, four held-out history blocks, eight material compositions, three known geometries and both excitation signs). Positive vertical values mean that the two-start selection has larger future position MSE than start 1. Horizontal values are the reduction in the past normalized squared-residual objective; the frozen selection chooses its strict floating-point minimum. Exact ties select start 1. The nonnegative horizontal axis and vertical axis use symmetric-log transforms, with linear thresholds 10⁻⁸ objective units and 0.001 mm², respectively. All cases, including the 5,027 coincident selections/forecasts at the origin, are plotted; overlap is substantial.

This is a presentation-only export made after the analysis lock. It changes language, axis bounds, typography and display units, not inputs, selections or statistical rules. Near-equal past likelihoods can accompany different future predictions: the median positive objective reduction is only 4.42 × 10⁻¹⁰. These results do not establish that every selected solution gives a materially better past fit or a worse forecast; the strong-probe aggregate is an explicit exception to the overall MSE increase. The fixed P1024 observer is not a converged posterior reference. Three banks and four history blocks support descriptive reporting here, not inference from 10,368 independent experiments.
"""
    (destination / "caption.md").write_text(caption)
    for path, expected in hashes.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Original artifact changed during rendering: {path}")
    provenance = {
        "role": "Post-lock presentation only; all original analysis outputs preserved",
        "script_sha256": sha(Path(__file__)),
        "source_manifest_sha256": sha(manifest_path),
        "input_sha256": {str(input_path): sha(input_path)},
        "original_analysis_artifacts_verified_before_and_after": len(hashes),
        "plotted_rows": plotted,
        "point_counts_by_probe_and_bank": {f"{p}/b{b}": int(n) for (p, b), n in frame.groupby(["probe", "bank"]).size().items()},
        "selection": "All input rows; none removed",
        "size_inches": [7, 3.35], "png_dpi": 300,
        "pdf_scatter_is_vector": True,
        "x_limits": [0, 1000], "y_limits_mm_squared": [-1000, 1000],
        "x_symlog_linthresh": 1e-8, "y_symlog_linthresh_mm_squared": .001,
        "output_sha256": {str(destination / n): sha(destination / n) for n in names if n != "provenance.json"},
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=DEST)
    args = parser.parse_args()
    print(json.dumps(export(args.source.resolve(), args.out.resolve()), indent=2))

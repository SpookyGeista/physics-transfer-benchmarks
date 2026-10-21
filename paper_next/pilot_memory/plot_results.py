"""Post-run presentation only: common y limits prevent roundoff magnification.

Original result CSVs and original renderings remain unchanged. No new inference.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pilot-memory-mpl")
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sources = [ROOT / "results/summary.csv", ROOT / "results/physical_contraction.csv", Path(__file__)]
    hashes = {str(p): sha(p) for p in sources}
    summary = pd.read_csv(sources[0]); physical = pd.read_csv(sources[1])
    dest = ROOT / "figures"
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("Existing final figures retained")
    dest.mkdir(exist_ok=True)
    metrics = [("initial_state_nrmse", "Alter Anfangszustand: NRMSE"),
               ("current_state_nrmse", "Zustand nach Probe: NRMSE"),
               ("future_force_rmse_N", "Zukünftige Kraft: RMSE [N]")]
    for kind in ["rest", "slip"]:
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
        for row, world in enumerate(["maxwell", "boucwen"]):
            for col, (metric, title) in enumerate(metrics):
                ax = axes[row, col]
                q = summary[(summary.world == world) & (summary.probe_kind == kind)]
                blind = q[(q.condition == "blinded") & (q.sigma_N == .05)]
                ax.plot(blind.duration_s, blind[metric], "ko--", ms=3, lw=1.8,
                        label="Kraft ausgeblendet; gleiche Probe")
                for sigma, color in [(.05, "#2166ac"), (.2, "#d95f02")]:
                    z = q[(q.condition == "sensed") & (q.sigma_N == sigma)]
                    ax.plot(z.duration_s, z[metric], "o-", color=color, ms=3,
                            label=f"Kraft beobachtet, σ={str(sigma).replace('.', ',')} N")
                values = summary[summary.world == world][metric].to_numpy()
                # Same axes for REST/SLIP; preserve values, including roundoff.
                ax.set_ylim(values.min() / 1.5, values.max() * 1.5)
                ax.set_yscale("log")
                ax.set_xscale("symlog", linthresh=.005)
                ax.set_xticks([0., .01, .1, 1.], ["0", "0,01", "0,1", "1"])
                ax.set_xlabel("Dauer der Probe [s]")
                ax.set_title(f"{world}: {title}", fontsize=10)
                ax.grid(alpha=.2, which="major")
                if col == 1:
                    ratio = physical[(physical.world == world) & (physical.probe_kind == kind)
                                     & (physical.duration_s == 1)].iloc[0].physical_spread_ratio
                    ax.text(.03, .06, f"Physische Spreizung nach 1 s:\n{ratio:.2g} × Anfangsspreizung",
                            transform=ax.transAxes, fontsize=8)
        axes[0, 0].legend(loc="lower left", fontsize=7)
        title = "RUHE: Die Kraftmessung liefert hier keine Zustandsinformation" if kind == "rest" else "SLIP: Messinformation und physische Rekonditionierung trennen"
        fig.suptitle(title, fontsize=14)
        fig.supxlabel("Gleiche y-Grenzen in Ruhe-/Slip-Diagrammen · Kraft-Forecast über 0,15 s · NRMSE bezogen auf ursprünglichen Priorbereich\n"
                      "Privilegierter Materialprüfstand mit vorgegebenem v/N · feste State-ID · deskriptive Mittel, keine Konfidenzintervalle",
                      fontsize=9)
        path = dest / f"memory_mechanism_{kind}.png"
        fig.savefig(path, dpi=180); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)
    assert hashes == {p: sha(p) for p in hashes}
    (dest / "provenance.json").write_text(json.dumps(dict(sources_sha256=hashes,
        outputs_sha256={p.name: sha(p) for p in dest.iterdir() if p.is_file()},
        role="post-run visualization with common axes only; no result or inferential rule changed"), indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()

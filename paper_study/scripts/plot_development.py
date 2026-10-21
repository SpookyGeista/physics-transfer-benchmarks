"""Static scientific figures from completed episode tables, no result filtering."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/paper-study-matplotlib")
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER=["none","hold_050","hold_150","hold_300","gentle_050","gentle_150","strong_150","strong_300"]
LABELS=["None","Hold\n50 ms","Hold\n150 ms","Hold\n300 ms","Probe 3 N\n50 ms*","Probe 3 N\n150 ms","Probe 8 N\n150 ms","Probe 8 N\n300 ms"]
COLORS={"prior_propagation":"#798b9d","trajectory_assimilation":"#147d92","privileged_full_history":"#df9542"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(args.data/"episodes.csv")
    data=data[data.method.str.startswith("joint_sensor_likelihood/")].copy()
    if "forecast_start" in data:
        data=data[data.forecast_start=="filtered_particles"]
    else:
        data["estimator"]=data.method.str.split("/").str[-1]
    specs=[("future_position_rmse_m",1000,"Future position RMSE [mm]","position"),
           ("future_friction_rmse_N",1,"Future friction RMSE [N]","friction")]
    v1="native_development" in str(args.data) and "v2" not in str(args.data)
    # An evaluation directory can have a different name: source lock identifies generation.
    lock=json.loads((args.data/"evaluation_lock.json").read_text())
    native_paths=list(lock.get("input_sha256",{}))
    v1=bool(native_paths) and not any("/p310" in s for s in native_paths)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    summary=data.groupby(["friction_world","sensor","probe","estimator"],observed=True)[[s[0] for s in specs]].mean()
    summary.to_csv(args.out/"summary.csv")
    for metric,scale,ylabel,stem in specs:
        fig,axes=plt.subplots(2,2,figsize=(13,7.6),sharex=True)
        for row,world in enumerate(["maxwell","boucwen"]):
            for col,sensor in enumerate(["ideal","noisy"]):
                ax=axes[row,col]
                for index,method in enumerate(COLORS):
                    values=np.array([summary.loc[(world,sensor,probe,method),metric]*scale for probe in ORDER])
                    ax.bar(np.arange(len(ORDER))+(index-1)*.25,values,.23,color=COLORS[method],
                        label={"prior_propagation":"Prior propagated", "trajectory_assimilation":"Trajectory likelihood", "privileged_full_history":"Privileged full history"}[method])
                title="Relaxing friction" if world=="maxwell" else "Aging friction"
                ax.set_title(title+" | "+("ideal sensors" if sensor=="ideal" else "noisy sensors"))
                ax.set_ylabel(ylabel); ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True)
                labels=LABELS if v1 else [s.replace("*","") for s in LABELS]
                ax.set_xticks(np.arange(len(ORDER)),labels,fontsize=8)
                ax.set_ylim(bottom=0)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,.98))
        fig.suptitle("Unknown history: additional trajectory assimilation",y=1.015,fontsize=16)
        footer="Exploratory averages; one fitted model bank, both normal laws. No confidence or generalization claim."
        if v1: footer+="\n*50-ms probe was sampled at sine zeros and is a degenerate hold; retained here."
        fig.text(.5,.01,footer,ha="center",fontsize=9)
        fig.tight_layout(rect=[0,.06,1,.94])
        for suffix in ("png","pdf"):
            fig.savefig(args.out/f"{stem}.{suffix}",dpi=180,bbox_inches="tight")
        plt.close(fig)
    provenance={"source_csv":str((args.data/"episodes.csv").resolve()),
        "source_sha256":hashlib.sha256((args.data/"episodes.csv").read_bytes()).hexdigest(),
        "rows":len(data),"v1_degenerate_probe":v1,"excluded_methods":"Conditional N-hold is separate diagnostic, not plotted",
        "uncertainty":"No CIs: development design and one model bank"}
    (args.out/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")

if __name__=="__main__":main()

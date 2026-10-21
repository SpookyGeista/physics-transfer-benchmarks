"""Constitutive illustration using the declared reachable equal-force pair."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/paper-matplotlib')
import sys,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'alternatives'))
from multistate_friction import reachable_equal_force_pair


def main():
    out=ROOT/'reports/figures/reachable_memory';out.mkdir(parents=True,exist_ok=True)
    a,b,meta=reachable_equal_force_pair('balanced')
    z=np.concatenate([a.audit()['deflection'],b.audit()['deflection']],axis=0)
    initial=np.array([a.observe()[0],b.observe()[0]])
    x=np.linspace(0,meta['diagnostic_future_displacement'],201);f=[[initial[0]],[initial[1]]]
    for dx in np.diff(x):
        for i,r in enumerate((a,b)):
            r.force(dx/.0025,15.,.0025);f[i].append(r.observe()[0])
    f=np.array(f)
    np.testing.assert_allclose(initial,[1.2,1.2],rtol=0,atol=1e-12)
    np.testing.assert_allclose(abs(f[0,-1]-f[1,-1]),.3,rtol=0,atol=1e-12)
    fig,axs=plt.subplots(1,3,figsize=(11.5,3.5));colors=('#087f88','#c36c23')
    for i,key in enumerate(('history_a','history_b')):
        h=meta[key];axs[0].plot(h['time'],h['x']*1e3,color=colors[i],label=f'History {"AB"[i]}',lw=2)
        axs[1].bar(np.arange(3)+(i-.5)*.32,z[i]*1e3,width=.3,color=colors[i],label=f'History {"AB"[i]}')
        axs[2].plot(x*1e3,f[i],color=colors[i],lw=2,label=f'History {"AB"[i]}')
    axs[0].set(title='Reachable preparation',xlabel='Fixture time [s]',ylabel='Displacement [mm]')
    axs[1].set(title='Different final memory',xlabel='Stop element',ylabel='Elastic deflection [mm]',xticks=[0,1,2],xticklabels=['1','2','3'])
    axs[2].set(title='Same future displacement',xlabel='Additional displacement [mm]',ylabel='Instantaneous friction force [N]')
    axs[0].legend(frameon=False,fontsize=9);axs[2].annotate('0.30 N difference',xy=(x[-1]*1e3,float(f.mean(axis=0)[-1])),xytext=(1.2,-2.),
        arrowprops=dict(arrowstyle='->',color='#475e65'),fontsize=9)
    for ax in axs:ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Identical x = v = 0, N = 15 N and current friction force = 1.2 N',fontweight='bold',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93))
    for ext in ('png','pdf'):fig.savefig(out/f'equal_present_different_future.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    np.savez_compressed(out/'illustration.npz',initial_force=initial,initial_deflection=z,future_displacement=x,future_force=f,
        history_a_x=meta['history_a']['x'],history_b_x=meta['history_b']['x'],history_time=meta['history_a']['time'])
    (out/'provenance.json').write_text(json.dumps(dict(role='Designed reachable constitutive example, not held-out algorithm evidence or native trajectory',
        parameters='balanced',sources_sha256={str(Path(__file__).resolve()):sha(__file__),str(ROOT/'alternatives/multistate_friction.py'):sha(ROOT/'alternatives/multistate_friction.py')},
        same_initial_force_N=initial.tolist(),future_force_difference_N=abs(float(f[0,-1]-f[1,-1])),
        outputs_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='provenance.json'}),indent=2)+'\n')


if __name__=='__main__':main()

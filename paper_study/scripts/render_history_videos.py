"""Animate fixed-index saved native trajectories and their causal forecasts.

This renders scientific plots, not an Isaac camera view. No inference is rerun.
Two fixed material pairs, r1/noisy calibration, noisy sensors, episode 7.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PAIRS=(('boucwen','boucwen'),('boucwen','maxwell'))
PROBES=('hold_150','gentle_150')
METHODS=('prior_propagation','trajectory_assimilation')
LABELS=('Prior propagation','Trajectory assimilation')
COLORS=('#c97d1b','#007f86')
EPISODE=7
FPS=25
SLOWDOWN=20


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_case(native,evaluation,seed,nw,fw):
    manifest=json.loads((evaluation/'artifact_manifest.json').read_text())
    data={};hashes={}
    for probe in PROBES:
        raw=native/f'p{seed}/{nw}_{fw}/{probe}/noisy.npz'
        pred=evaluation/f'r1/noisy/p{seed}/{nw}_{fw}/{probe}/noisy/joint_sensor_likelihood.npz'
        metadata=json.loads(raw.with_suffix('.json').read_text())
        if sha(raw)!=metadata['npz_sha256']:raise ValueError('Native hash mismatch')
        if sha(pred)!=manifest['outputs_sha256'][str(pred.resolve())]:raise ValueError('Forecast hash mismatch')
        with np.load(raw,allow_pickle=False) as z:
            dt=float(metadata['dt']);boundary=int(z['probe_steps'])
            state=np.concatenate([z['state'][EPISODE,:1],z['next_state'][EPISODE]],axis=0)
            force=z['friction'][EPISODE].copy()
        with np.load(pred,allow_pickle=False) as z:
            forecasts={method:z[f'filtered_particles_{method}_forecast'][EPISODE].copy() for method in METHODS}
        data[probe]=dict(dt=dt,boundary=boundary,state=state,force=force,forecasts=forecasts)
        for path in (raw,raw.with_suffix('.json'),pred,evaluation/'artifact_manifest.json'):
            hashes[str(path.resolve())]=sha(path)
    return data,hashes


def render(data,path,nw,fw,seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,2,figsize=(12.8,7.2),layout='constrained',sharex=True,sharey='row')
    dynamic=[];cursors=[]
    for col,probe in enumerate(PROBES):
        d=data[probe];dt=d['dt'];b=d['boundary'];total=(len(d['state'])-1)*dt
        ts=np.arange(len(d['state']))*dt;tf=np.arange(len(d['force']))*dt
        future_t=(b+1+np.arange(len(d['state'])-b-1))*dt
        force_future_t=future_t-dt
        axes[0,col].set_title('Hold for 150 ms' if col==0 else '3 N probe for 150 ms',weight='bold')
        for row in range(3):
            ax=axes[row,col];ax.axvline(b*dt,color='#888',ls='--',lw=1)
            ax.axvspan(0,b*dt,color='#dce5ed',alpha=.5);ax.grid(alpha=.17);ax.set_xlim(0,total)
            cursors.append(ax.axvline(0,color='#383838',lw=.8))
        for row,t,y in ((0,ts,d['state'][:,0]*1000),(1,tf,d['force'])):
            line=axes[row,col].plot([],[],color='#222',lw=2,label='Native PhysX')[0]
            dynamic.append((line,t,y,False))
        for method,label,color in zip(METHODS,LABELS,COLORS):
            forecast=d['forecasts'][method]
            for row,t,y in ((0,future_t,forecast[:,0]*1000),
                            (1,force_future_t,forecast[:,3]),
                            (2,future_t,(forecast[:,0]-d['state'][b+1:,0])*1e6)):
                line=axes[row,col].plot([],[],color=color,lw=1.7,label=label)[0]
                dynamic.append((line,t,y,True))
        axes[2,col].axhline(0,color='#555',lw=.7)
        axes[2,col].set_xlabel('Simulation time [s]')
    axes[0,0].set_ylabel('Position [mm]');axes[1,0].set_ylabel('Friction force [N]')
    axes[2,0].set_ylabel('Prediction error [µm]')
    # Fixed limits from all stored curves prevent changing scales during playback.
    for row in range(3):
        values=np.concatenate([y for line,t,y,p in dynamic if line.axes in axes[row]])
        lo,hi=float(values.min()),float(values.max());pad=max((hi-lo)*.12,[.01,.05,2][row])
        axes[row,0].set_ylim(lo-pad,hi+pad)
    axes[0,0].legend(loc='upper left',fontsize=8,framealpha=.9)
    title=fig.suptitle('',fontsize=13,weight='bold')
    fig.supxlabel('Animated plots of saved native PhysX data · no camera render · 20× slow motion\n'
                  'Forecasts use observations through 150 ms only; all model parameters remain fixed.',fontsize=10)
    duration=max((len(d['state'])-1)*d['dt'] for d in data.values())
    times=np.linspace(0,duration,round(duration*SLOWDOWN*FPS)+1)
    def update(now):
        for line,t,y,prediction in dynamic:
            # Reveal the entire committed future forecast at the forecast boundary.
            keep=np.ones(len(t),dtype=bool) if prediction and now>=.15 else t<=now
            if prediction and now<.15:keep[:]=False
            line.set_data(t[keep],y[keep])
        for cursor in cursors:cursor.set_xdata([now,now])
        title.set_text(f'Unknown material history · N: {nw}, friction: {fw}\n'
                       f'Fixed case: history block {seed}, episode 7, model r1/noisy · t = {now:.3f} s')
    writer=FFMpegWriter(fps=FPS,codec='libx264',bitrate=2800,
                       extra_args=['-pix_fmt','yuv420p','-movflags','+faststart'])
    with writer.saving(fig,str(path),dpi=100):
        for now in times:update(now);writer.grab_frame()
    update(duration);fig.savefig(path.with_suffix('.png'),dpi=150)
    plt.close(fig)
    return dict(frames=len(times),fps=FPS,simulation_seconds=duration,slowdown=SLOWDOWN)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path,default=ROOT/'results/native_development_v2')
    parser.add_argument('--evaluation',type=Path,default=ROOT/'results/observer_development_v2')
    parser.add_argument('--history-seed',type=int,default=31001)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError('Existing videos retained; choose fresh output')
    if not out.is_relative_to(ROOT):raise ValueError('Output must stay inside paper_study')
    out.mkdir(parents=True)
    for nw,fw in PAIRS:
        data,hashes=load_case(args.native.resolve(),args.evaluation.resolve(),args.history_seed,nw,fw)
        path=out/f'history_{nw}_{fw}.mp4'
        detail=render(data,path,nw,fw,args.history_seed)
        if any(sha(p)!=h for p,h in hashes.items()):raise ValueError('Input changed during rendering')
        path.with_suffix('.json').write_text(json.dumps(dict(**detail,episode=EPISODE,
            history_seed=args.history_seed,selection='Fixed numeric index, not selected by error',
            representation='Animated scientific plots of stored native trajectories; not an Isaac RGB camera render',
            sources_sha256=hashes|{str(Path(__file__).resolve()):sha(__file__)},
            video_sha256=sha(path)),indent=2)+'\n')
        print(path,flush=True)


if __name__=='__main__':main()

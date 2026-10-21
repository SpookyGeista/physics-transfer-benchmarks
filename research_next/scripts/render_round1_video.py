"""Fixed-case visualization of saved native PhysX trajectories, not RGB rendering.

Two predeclared crossed compositions, r1/clean/episode0, four fixed methods.
No best-case selection, no inference, no simulator or robot connection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/round1'
METHODS=('recurrent','blind_state_id','path_ridge','nominal')
COMPOSITIONS=(('maxwell','boucwen'),('boucwen','maxwell'))
LABELS={'recurrent':'Lokale GRU','blind_state_id':'Lokale Familienwahl','path_ridge':'Verlaufs-Ridge','nominal':'Nominal'}
COLORS=('#2267a2','#087f6c','#dc8318','#737373')
FPS=30
SLOWDOWN=1.5


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timeline(test,controller,episode,dt):
    """States are sampled at native state times; forces use pre-update timestamps."""
    warm=np.asarray(test['warm_state'])[episode]
    initial=np.asarray(controller['state'])[episode,0:1]
    state=np.concatenate((warm,initial,np.asarray(controller['next_state'])[episode]),axis=0)
    normal=np.r_[test['warm_normal'][episode],controller['normal'][episode],controller['normal'][episode,-1]]
    delta=np.r_[test['warm_delta'][episode],controller['delta'][episode],controller['delta'][episode,-1]]
    target=np.r_[np.zeros(len(warm)),controller['target'][episode],controller['target'][episode,-1]]
    t=np.arange(len(state))*dt
    if not (len(normal)==len(delta)==len(target)==len(t)):raise ValueError('Mismatched native state/force timeline')
    if not all(np.isfinite(x).all() for x in (state,normal,delta,target)):raise ValueError('Nonfinite fixed-case video data')
    return dict(time=t,state=state,normal=normal,delta=delta,target=target,warm_seconds=len(warm)*dt)


def load_fixed(root,nw,fw,cfg):
    testpath=root/f'test/{nw}_{fw}/r1.npz'
    meta=json.loads(testpath.with_suffix('.json').read_text())
    if sha(testpath)!=meta['npz_sha256']:raise ValueError('Test artifact hash differs')
    with np.load(testpath) as z:test={k:z[k] for k in ('warm_state','warm_normal','warm_delta','amplitude')}
    hashes={str(testpath):sha(testpath),str(testpath.with_suffix('.json')):sha(testpath.with_suffix('.json'))}
    data={}
    for method in METHODS:
        path=root/f'controllers/{nw}_{fw}/r1/clean/{method}.npz'
        md=json.loads(path.with_suffix('.json').read_text())
        if md['rows']!=meta['rows'] or md['method']!=method:raise ValueError('Video row/method mismatch')
        if md['noise']!='clean' or md['seed']!=1:raise ValueError('Only fixed clean r1 is permitted')
        if sha(path)!=md['npz_sha256']:raise ValueError('Controller artifact hash differs')
        with np.load(path) as z:controller={k:z[k] for k in ('state','next_state','normal','delta','target')}
        data[method]=timeline(test,controller,0,cfg['dt'])
        hashes.update({str(path):sha(path),str(path.with_suffix('.json')):sha(path.with_suffix('.json'))})
    return data,float(test['amplitude'][0]),meta['rows'][0],hashes


def create_figure(data,amplitude,nw,fw):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(12.8,7.2),layout='constrained')
    gs=fig.add_gridspec(3,2,width_ratios=(1.1,1.7))
    schematic=fig.add_subplot(gs[:,0])
    axes=[fig.add_subplot(gs[i,1]) for i in range(3)]
    allx=np.concatenate([v['state'][:,0] for v in data.values()])
    xmin=min(-.045,float(allx.min())-.012);xmax=max(.045,float(allx.max())+.012)
    track=np.linspace(xmin,xmax,401)
    boxes=[];status=[]
    for i,(method,color) in enumerate(zip(METHODS,COLORS)):
        base=(3-i)*1.2
        schematic.plot(track,np.full_like(track,base),color='#bac3ca',lw=2)
        # Geometry curve is illustrative; only slider horizontal position is SI-scaled.
        schematic.plot(track,base+.35+30*amplitude*np.tanh(track/.04),color='#b99c44',lw=2)
        box=Rectangle((-.004,base+.035),.008,.20,color=color,alpha=.9)
        schematic.add_patch(box);boxes.append((box,base))
        schematic.text(xmin,base+.81,LABELS[method],color=color,weight='bold',va='center')
        status.append(schematic.text(xmin,base+.58,'',fontsize=8,va='center'))
    target_marker=schematic.axvline(0,color='black',linestyle='--',lw=.9,alpha=.55)
    schematic.set_xlim(xmin,xmax);schematic.set_ylim(-.2,4.6)
    schematic.set_yticks([]);schematic.set_xlabel('Horizontale Schlittenposition x [m]')
    schematic.set_title('Schlitten + Nocke: schematische Geometrie\nVertikalzeichnung überhöht; x maßstäblich')
    lines=[[],[],[]]
    for method,color in zip(METHODS,COLORS):
        for i in range(3):lines[i].append(axes[i].plot([],[],color=color,label=LABELS[method],lw=1.7)[0])
    t=next(iter(data.values()))['time']
    for ax in axes:
        ax.set_xlim(0,float(t[-1]));ax.axvline(4,color='#777777',ls=':',lw=1)
        ax.axvline(4.6,color='#bbbbbb',ls=':',lw=.8);ax.grid(alpha=.18)
    axes[0].plot(t,next(iter(data.values()))['target']*1000,'k--',lw=1,label='Ziel',drawstyle='steps-post')
    fields=[np.concatenate([d['state'][:,0]*1000 for d in data.values()]),
            np.concatenate([d['state'][:,1] for d in data.values()]),
            np.concatenate([d['normal'] for d in data.values()])]
    for ax,values in zip(axes,fields):
        lower=float(values.min());upper=float(values.max());pad=max((upper-lower)*.15,.1 if ax==axes[0] else .01)
        if ax==axes[0]:lower=min(lower,-35);upper=max(upper,35)
        ax.set_ylim(lower-pad,upper+pad)
    axes[0].set_ylabel('Position [mm]');axes[1].set_ylabel('Geschwindigkeit [m/s]');axes[2].set_ylabel('Normalkraft [N]')
    axes[2].set_xlabel('Simulationszeit inklusive Vorbereitung [s]')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,1.30),ncol=3,fontsize=8,frameon=False)
    cursors=[ax.axvline(0,color='#333333',lw=.8) for ax in axes]
    heading=fig.suptitle('',fontsize=12,weight='bold')
    fig.supxlabel('Visualisierung gespeicherter nativer PhysX-Trajektorien · kein RGB-Simulationsrender · 1,5× Zeitlupe',fontsize=10)

    def update(simtime):
        idx=min(int(np.searchsorted(t,simtime,side='right')),len(t))
        for j,(method,color) in enumerate(zip(METHODS,COLORS)):
            d=data[method]
            x=float(np.interp(simtime,t,d['state'][:,0]));v=float(np.interp(simtime,t,d['state'][:,1]));normal=float(np.interp(simtime,t,d['normal']))
            boxes[j][0].set_x(x-.004)
            status[j].set_text(f'x={x*1000:+.2f} mm   v={v:+.3f} m/s   N={normal:.2f} N')
            lines[0][j].set_data(t[:idx],d['state'][:idx,0]*1000)
            lines[1][j].set_data(t[:idx],d['state'][:idx,1])
            lines[2][j].set_data(t[:idx],d['normal'][:idx])
        target=float(next(iter(data.values()))['target'][max(0,idx-1)])
        target_marker.set_xdata([target,target])
        for cursor in cursors:cursor.set_xdata([simtime,simtime])
        phase='Vorbereitung: Rahmen ideal eingespannt' if simtime<4 else 'Freie Bewegung mit MPC · Zielwechsel bei 4,6 s'
        heading.set_text(f'N: {nw}   Ft: {fw} · r1 / clean / Episode 0\n{phase} · t={simtime:0.3f} s')
        return fig
    update(0)
    return fig,update


def render(data,amplitude,nw,fw,path):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    if not FFMpegWriter.isAvailable():
        # Reuse the bundled binary already used by the prior study's imageio.
        import imageio_ffmpeg
        matplotlib.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
    if not FFMpegWriter.isAvailable():raise RuntimeError('Installed ffmpeg is required; do not substitute a fake video')
    fig,update=create_figure(data,amplitude,nw,fw)
    writer=FFMpegWriter(fps=FPS,codec='libx264',bitrate=3500,extra_args=['-pix_fmt','yuv420p','-movflags','+faststart'])
    duration=float(next(iter(data.values()))['time'][-1])
    times=np.arange(round(duration*SLOWDOWN*FPS)+1)/(SLOWDOWN*FPS)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(f'Fixed-case video already exists: {path}')
    try:
        with writer.saving(fig,str(path),dpi=110):
            for simtime in times:
                update(min(simtime,duration));writer.grab_frame()
        update(duration);fig.savefig(path.with_suffix('.png'),dpi=150,bbox_inches='tight')
    finally:plt.close(fig)
    return dict(frames=len(times),fps=FPS,simulated_seconds=duration,playback_seconds=len(times)/FPS,slowdown=SLOWDOWN)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    sys.path.insert(0,str(ROOT/'src'))
    from round1_lock import verify_test_lock,verify_model_lock
    lock=dict(test_lock_sha256=verify_test_lock(),model_lock_sha256=verify_model_lock())
    cfg=json.loads((ROOT/'configs/round1.json').read_text())
    dest=ROOT/'videosround1';dest.mkdir(parents=True,exist_ok=True)
    for nw,fw in COMPOSITIONS:
        data,amplitude,row,hashes=load_fixed(OUT,nw,fw,cfg)
        path=dest/f'round1_{nw}_{fw}_r1_clean_episode0.mp4'
        details=render(data,amplitude,nw,fw,path)
        meta=dict(**details,**lock,normal_world=nw,friction_world=fw,seed=1,noise='clean',episode=0,
                  methods=list(METHODS),fixed_before_test=True,row=row,amplitude=amplitude,
                  representation='Schematic visualization of recorded native PhysX states and applied reference forces; not an RGB simulator render',
                  source_sha256=hashes|{str(Path(__file__)):sha(Path(__file__)),str(ROOT/'configs/round1.json'):sha(ROOT/'configs/round1.json')},
                  video_sha256=sha(path))
        path.with_suffix('.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
        print('ROUND1_VIDEO',path,details,flush=True)
    if dict(test_lock_sha256=verify_test_lock(),model_lock_sha256=verify_model_lock())!=lock:
        raise RuntimeError('Model/test lock changed during visualization')


if __name__=='__main__':main()

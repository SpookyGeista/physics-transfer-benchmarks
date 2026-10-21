"""Prespecified single-case trajectory animation, not camera/robot footage."""
from pathlib import Path
import os,sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/physics-transfer-matched-matplotlib')
sys.dont_write_bytecode=True
import argparse,json,time,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    with Path(p).open('w') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')

def main(data,out):
    if out.exists():raise FileExistsError('Fresh export directory required')
    if json.loads((data/'process.json').read_text())['status']!='complete':raise ValueError('Complete study required')
    lock=json.loads((data/'lock.json').read_text());manifest=json.loads((data/'manifest.json').read_text())
    if sha(Path(__file__))!=lock['source_sha256'][str(Path(__file__).resolve())]:raise ValueError('Changed video source')
    sources=dict(lock['source_sha256']);sources.update({str(data/p):sha(data/p) for p in ['process.json','lock.json','manifest.json']})
    for p,h in sources.items():
        if sha(p)!=h:raise ValueError('Changed source/input: '+p)
    selected=[]
    for duration in (150,300):
        for strength in (0,3,12):
            p=data/'batches/bank1_boucwen_balanced/h71001'/f's+1_d{duration}_a{strength}.npz'
            if sha(p)!=manifest['outputs_sha256'][str(p)]:raise ValueError('Changed selected batch')
            sources[str(p)]=sha(p)
            with np.load(p,allow_pickle=False) as f:
                d={k:f[k].copy() for k in f.files}
            m=json.loads(str(d['metadata_json']));e=int(np.flatnonzero(d['profile_indices']==1)[0])
            ni=m['sensors'].index('noisy');fi=m['futures'].index(10.)
            selected.append((duration,strength,d,m,e,ni,fi))
    out.mkdir(parents=True);started=time.perf_counter();write(out/'process.json',{'status':'running','started_unix':time.time()})
    try:
        fig,axes=plt.subplots(2,3,figsize=(12.8,7.2),dpi=100,sharex=True,sharey=True)
        t=np.arange(1,161)*.0025;artist=[];ymin=np.inf;ymax=-np.inf
        for ax,(duration,strength,d,m,e,ni,fi) in zip(axes.ravel(),selected):
            truth=d['truth_future'][fi,e,:,0]*1000
            prior=d['predictions'][ni,fi,e,m['methods'].index('prior'),:,0]*1000
            post=d['predictions'][ni,fi,e,m['methods'].index('posterior'),:,0]*1000
            mle=d['predictions'][ni,fi,e,m['methods'].index('mle_two_start'),:,0]*1000
            # A fixed full-trajectory range prevents animated auto-axis changes.
            allvalues=np.r_[truth,prior,post,mle];span=max(np.ptp(allvalues),1.)
            ymin=min(ymin,-1.,allvalues.min()-.08*span);ymax=max(ymax,1.,allvalues.max()+.08*span)
            ax.set_xlim(0,.4);ax.grid(alpha=.18);ax.set_title(f'{duration} ms preparation; {strength} N')
            lines=[]
            for values,color,style,label in [(truth,'black','-','Synthetic truth'),(prior,'#888888',':','Uniform prior'),
                (post,'#0072B2','-','Conditioned particles'),(mle,'#D55E00','--','Two-start MLE')]:
                line,=ax.plot([],[],style,color=color,lw=1.8,label=label);lines.append((line,values))
            cursor=ax.axvline(0,color='#888888',alpha=.45,lw=.8);artist.append((lines,cursor))
        axes[0,0].set_ylim(ymin,ymax)
        for ax in axes[-1]:ax.set_xlabel('Future time (s)')
        for ax in axes[:,0]:ax.set_ylabel('Position (mm)')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,.96),frameon=False)
        fig.suptitle('Same reachable history, six preparation programs',y=.995)
        clock=fig.text(.5,.02,'',ha='center',fontsize=10)
        fig.text(.5,.055,'Fixed illustrative case: bank 1 / Bouc-Wen + balanced PI / history 71001:1 / noisy / future 10 N',ha='center',fontsize=9)
        fig.subplots_adjust(top=.85,bottom=.16,hspace=.3,wspace=.15)
        destination=out/'matched_preparation_comparison.mp4'
        with imageio.get_writer(destination,fps=20,codec='libx264',quality=8,macro_block_size=16) as writer:
            for frame in range(160):
                for lines,cursor in artist:
                    for line,values in lines:line.set_data(t[:frame+1],values[:frame+1])
                    cursor.set_xdata([t[frame],t[frame]])
                clock.set_text(f'Simulation time {t[frame]:.4f} s | 20x slow motion | trajectory plot, not camera footage')
                fig.canvas.draw();writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:,:,:3])
            fig.savefig(out/'poster.png',dpi=100)
        plt.close(fig)
        for p,h in sources.items():
            if sha(p)!=h:raise ValueError('Input changed during export: '+p)
        write(out/'manifest.json',dict(input_sha256=sources,outputs_sha256={str(p):sha(p) for p in [destination,out/'poster.png']},
            selection='Specified in source before main data: bank1,boucwen,balanced,history71001,profile1,sign+1,noisy,future10N,all6arms',
            frames=160,fps=20,seconds=8.,simulated_seconds=.4,scope='Fixed-case scientific trajectory animation; no robot or camera footage'))
        write(out/'process.json',dict(status='complete',seconds=time.perf_counter()-started,frames=160))
        print('VIDEO COMPLETE',destination)
    except BaseException as exc:
        plt.close('all');write(out/'process.json',dict(status='failed',error=repr(exc),seconds=time.perf_counter()-started));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.data.resolve(),a.out.resolve())

"""Scientific schematic from the declared protocol; not a simulator screenshot."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/paper-matplotlib')
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]


def box(ax,x,y,w,h,text,color='#edf3f5'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.014',linewidth=1.2,
        edgecolor='#28464e',facecolor=color))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,linespacing=1.6)


def arrow(ax,a,b,label=None,offset=(0,0)):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#28464e',lw=1.4))
    if label:ax.text((a[0]+b[0])/2+offset[0],(a[1]+b[1])/2+offset[1],label,fontsize=9,ha='center',va='center',bbox=dict(facecolor='white',edgecolor='none',pad=2))


def main():
    out=ROOT/'reports/figures/design';out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,1,figsize=(7.2,9.4))
    for ax in axes:ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    a,b=axes
    a.set_title('A. Known coupling, fixed learned laws',loc='left',fontweight='bold',fontsize=12)
    box(a,.26,.77,.48,.13,'Normal memory model\nN = fₙ(δ, δ̇, state)')
    box(a,.03,.43,.37,.17,'Friction memory model\nFₜ = fₜ(v, N, state)', '#faedda')
    box(a,.60,.43,.37,.17,'Native mass: 2 kg\nx, v', '#deeff0')
    arrow(a,(.28,.77),(.22,.60),'N',(.015,0))
    arrow(a,(.40,.51),(.60,.51),'Fₜ',(0,.045))
    arrow(a,(.71,.77),(.82,.62),'−g′(x)N',(.03,.015))
    arrow(a,(.60,.435),(.40,.435),'v',(0,-.035))
    arrow(a,(.77,.18),(.77,.42),'u',(0,0))
    a.text(.5,.98,'d',ha='center',va='center',fontsize=10)
    arrow(a,(.5,.95),(.5,.91))
    a.text(.5,.28,'δ = d + g(x)\nδ̇ = ḋ + g′(x)v',ha='center',va='center',fontsize=11)
    a.text(.5,.075,'Two external work ports: u·v and N·ḋ\nHidden history; known geometry; no online parameter fit',ha='center',va='center',fontsize=9)
    b.set_title('B. History and causal information',loc='left',fontweight='bold',fontsize=12)
    box(b,.025,.76,.28,.15,'4 s ideal fixture\nReachable history\nHidden from estimator','#faedda')
    box(b,.37,.76,.24,.15,'Native probe\nCausal x / v / N\nFixed local fits')
    box(b,.68,.76,.29,.15,'Future program\nNo sensor updates\nScore saved forecasts','#deeff0')
    arrow(b,(.31,.835),(.355,.835));arrow(b,(.615,.835),(.675,.835))
    b.text(.03,.64,'Decision experiment: every branch has the same first 50 ms',fontsize=10,fontweight='bold')
    box(b,.025,.34,.29,.16,'Common 50 ms prefix\nOld-rate N sample\nChoose using prefix only')
    box(b,.59,.49,.36,.09,'Release now')
    box(b,.59,.32,.36,.09,'Hold for another 100 ms')
    box(b,.59,.15,.36,.09,'Continue probe for 100 ms')
    for y in (.535,.365,.195):arrow(b,(.33,.42),(.575,y))
    b.text(.63,.085,'All future commands fixed at the choice;\nlater probe samples refine forecasts only.',ha='center',va='center',fontsize=9)
    fig.subplots_adjust(left=.06,right=.965,bottom=.04,top=.92,hspace=.24)
    fig.suptitle('Prediction after unknown material history',fontsize=15,fontweight='bold',y=.98)
    for ext in ('png','pdf'):fig.savefig(out/f'experimental_design.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'provenance.json').write_text(json.dumps(dict(role='Protocol schematic, not camera footage',
        source_sha256=sha(Path(__file__)),outputs={p.name:sha(p) for p in out.glob('experimental_design.*')}),indent=2)+'\n')


if __name__=='__main__':main()

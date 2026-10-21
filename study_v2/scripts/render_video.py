"""Isaac Sim rendering of saved trajectories, explicitly labeled data replay."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--world',default='maxwell');p.add_argument('--episode',type=int,default=0);p.add_argument('--task',choices=['replay','control'],default='replay');p.add_argument('--study',choices=['original','supplement'],default='original');AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera,CameraCfg
from pxr import UsdGeom,Gf
from decision_study.common import DT,CFG,dump,digest

sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/30,device='cpu',gravity=(0.,0.,0.)))
sim_utils.DomeLightCfg(intensity=1800).func('/World/Light',sim_utils.DomeLightCfg(intensity=1800))
methods=['reference','nominal','analytic_flex','modular','joint','moe'] if a.task=='replay' else ['identified','nominal','analytic_flex','modular','joint','moe'];colors=[(.1,.7,.6),(.7,.25,.2),(.85,.6,.15),(.55,.35,.8),(.1,.45,.9),(.6,.65,.7)]
prefix='supplement_exact/' if a.study=='supplement' and a.task=='replay' else ''
if a.task=='replay':
    truth=np.load(f'{prefix}results/data/{a.world}/r0/replay.npz');trajs={m:truth['next_state'][a.episode] if m=='reference' else np.load(f'{prefix}results/traces/{a.world}_{m}.npz')['state'][a.episode] for m in methods}
    compression=truth['delta'][a.episode]
else:
    trajs={m:np.load(f'results/control/{a.world}_r0_clean_{m}.npz')['next_state'][a.episode] for m in methods}
    compression=np.full(len(trajs[methods[0]]),.02)
length=len(trajs[methods[0]])
poses=[];pads=[]
for i,m in enumerate(methods):
    rail=sim_utils.CuboidCfg(size=(3.4,.16,.035),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.15,.18,.22)))
    rail.func(f'/World/Rail{i}',rail,translation=(0.,i*.3,0.))
    cfg=sim_utils.CuboidCfg(size=(.12,.1,.09),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colors[i]))
    cfg.func(f'/World/Slider{i}',cfg,translation=(0.,i*.3,.065))
    prim=sim.stage.GetPrimAtPath(f'/World/Slider{i}');poses.append(next(op for op in UsdGeom.Xformable(prim).GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeTranslate))
    pcfg=sim_utils.CuboidCfg(size=(.10,.1,.035),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.9,.75,.3)))
    pcfg.func(f'/World/Pad{i}',pcfg,translation=(0.,i*.3,.2));pads.append(next(op for op in UsdGeom.Xformable(sim.stage.GetPrimAtPath(f'/World/Pad{i}')).GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeTranslate))
cam=Camera(CameraCfg(prim_path='/World/Camera',update_period=0,height=720,width=1280,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=20.,focus_distance=3.,horizontal_aperture=24.,clipping_range=(.1,10.))))
sim.reset();cam.set_world_poses_from_view(torch.tensor([[0.,-.9,3.2]]),torch.tensor([[0.,.75,.02]]))
for _ in range(6):sim.render();cam.update(1/30)
try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
except OSError:font=ImageFont.load_default();small=font
out=Path('videos');out.mkdir(exist_ok=True)
with imageio.get_writer(out/f'{a.world}_{a.task}.mp4',fps=30,codec='libx264',quality=8,macro_block_size=16) as writer:
    for frame in range(120):
        t=min(int(frame/120*(length-1)),length-1)
        for i,m in enumerate(methods):
            x=float(trajs[m][t,0]);poses[i].Set(Gf.Vec3d(2*x,i*.3,.065));pads[i].Set(Gf.Vec3d(2*x,i*.3,.18-min(.1,max(0,float(compression[t])*3))))
        sim.render();cam.update(1/30)
        rgb=cam.data.output['rgb'][0,:,:,:3].cpu().numpy();im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,1280,86),fill=(14,22,32));draw.text((24,14),'DECISION STUDY V2 | Isaac Sim recorded trajectory replay',font=font,fill='white');draw.text((24,48),f'{a.world} / {a.task} / {a.study} | episode {a.episode}, seed 0 | t={t*DT:.3f}s | motion 2x, pad 3x',font=small,fill=(180,200,220))
        for i,m in enumerate(methods):
            c=tuple(int(255*x) for x in colors[i]);draw.rectangle((15,100+30*i,26,119+30*i),fill=c);draw.text((36,100+30*i),f'{m:12s} x={trajs[m][t,0]:+.3f} m',font=small,fill=c)
        draw.rectangle((0,674,1280,720),fill=(14,22,32));draw.text((24,686),'Recorded native PhysX / frozen model data. Visual replay; no robot or hardware.',font=small,fill='white')
        writer.append_data(np.asarray(im))
        if frame in (30,70):im.save(out/f'{a.world}_{a.task}_frame_{frame}.png')
dump(out/f'{a.world}_{a.task}.json',dict(world=a.world,study=a.study,task=a.task,episode=a.episode,position_visual_scale=2,selection='fixed episode0/replicate0, predeclared',frames=120,fps=30,source='Saved measured reference and predicted trajectories',render_backend='Isaac Sim RTX',physics_rerun=False,pad_visual_scale=3))
print('VIDEO_SAVED',out/f'{a.world}_{a.task}.mp4',flush=True)
app.close(wait_for_replicator=False,skip_cleanup=True)

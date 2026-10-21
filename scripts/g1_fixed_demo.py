"""Supplementary frozen-model load prediction on an installed fixed-base G1.
Existing Unitree articulation/implicit actuator configuration, simulation only.
No manufacturer policy claim; no SDK, DDS, ROS or hardware data.
"""
import argparse,os,sys,importlib.util,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--render',action='store_true');AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from pxr import UsdPhysics
from null_physics_transfer.reference import Reference
from null_physics_transfer.predict import load_model
from null_physics_transfer.common import DT,features,history_update,dump,digest

os.environ['PROJECT_ROOT']='/home/loya2/unitree_sim_isaaclab'
config_path=Path(os.environ['PROJECT_ROOT'])/'robots/unitree.py';spec=importlib.util.spec_from_file_location('installed_unitree_robot_config',config_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
cfg=module.G129_CFG_WITH_DEX3_BASE_FIX.copy();cfg.prim_path='/World/G1';cfg.spawn.articulation_props.fix_root_link=True
sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT,device='cpu'))
robot=Articulation(cfg)
# Pair interaction is exclusively the programmed external material load.
for prim in sim.stage.Traverse():
 if prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
if a.render:
 from isaaclab.sensors import Camera,CameraCfg
 sim_utils.DomeLightCfg(intensity=1200).func('/World/Light',sim_utils.DomeLightCfg(intensity=1200))
 camera=Camera(CameraCfg(prim_path='/World/Camera',height=720,width=960,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=22.,horizontal_aperture=24.,clipping_range=(.1,10.))))
sim.reset();body_ids,body_names=robot.find_bodies('left_wrist_yaw_link');joint_ids,joint_names=robot.find_joints(['left_elbow_joint','left_shoulder_pitch_joint'],preserve_order=True)
assert len(body_ids)==1 and len(joint_ids)==2,(robot.body_names,robot.joint_names)
b=body_ids[0];target=robot.data.default_joint_pos.clone();target[:,joint_ids[0]]=.6;target[:,joint_ids[1]]=-.25
limits=robot.data.soft_joint_pos_limits[0,joint_ids].cpu().numpy();assert limits[0,0]<.3 and limits[0,1]>1.,limits
for _ in range(400):
 robot.set_joint_position_target(target);robot.write_data_to_sim();sim.step(render=False);robot.update(DT)
xbase=float(robot.data.body_pos_w[0,b,0]);ref=Reference('sls',1,np.array([.014]));h=np.zeros((1,3,2));h[:,:,0]=.014
models={m:load_model(Path('results/models/sls/b24/s0')/m) for m in ['nominal','identified','analytic','joint','moe','modular']};records=[];preds={m:[] for m in models};q=[];desired=[];video=None
if a.render:
 import imageio.v2 as imageio
 from PIL import Image,ImageDraw,ImageFont
 try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
 except OSError:font=ImageFont.load_default()
 camera.set_world_poses_from_view(torch.tensor([[2.,2.,1.6]]),torch.tensor([[0.,0.,.8]]))
 for _ in range(6):sim.render();camera.update(DT)
 video=imageio.get_writer('videos/g1_fixed_load_demo.mp4',fps=25,codec='libx264',quality=8,macro_block_size=16)
start=time.time()
for t in range(1600):
 seconds=t*DT;target[:,joint_ids[0]]=.65+.30*np.sin(2*np.pi*.5*seconds);target[:,joint_ids[1]]=-.25+.08*np.sin(2*np.pi*.25*seconds)
 x=np.array([float(robot.data.body_pos_w[0,b,0])-xbase]);v=np.array([float(robot.data.body_lin_vel_w[0,b,0])]);delta=.014+.04*x;rate=.04*v
 n,ft=ref.observe_force(delta,rate,v);fx=ft-.04*n
 f=features(delta,rate,v,h,n)
 for m,model in models.items():pn,pf=model.predict(f);preds[m].append([float(pn[0]),float(pf[0])])
 load=torch.tensor([[[float(fx[0]),0.,0.]]]);robot.set_external_force_and_torque(load,torch.zeros_like(load),body_ids=body_ids,is_global=True)
 robot.set_joint_position_target(target);robot.write_data_to_sim();sim.step(render=False);robot.update(DT)
 records.append([seconds,float(x[0]),float(v[0]),float(delta[0]),float(n[0]),float(ft[0]),float(fx[0])]);q.append(robot.data.joint_pos[0,joint_ids].cpu().numpy().copy());desired.append(target[0,joint_ids].cpu().numpy().copy());h=history_update(h,delta,v)
 if a.render and t%16==0:
  sim.render();camera.update(16*DT);im=Image.fromarray(camera.data.output['rgb'][0,:,:,:3].cpu().numpy());draw=ImageDraw.Draw(im)
  draw.rectangle((0,0,960,80),fill=(12,21,30));draw.text((18,10),'G1 | FIXED BASE | simulated material load at left wrist',font=font,fill='white');draw.text((18,39),f't={seconds:.2f}s  N={n[0]:.2f} N  F_t={ft[0]:.2f} N  | existing implicit actuator gains',font=font,fill=(150,200,230))
  draw.rectangle((0,662,960,720),fill=(12,21,30));draw.text((18,674),'No whole-body stability claim. No hardware / SDK. Supplemental trial.',font=font,fill='white');video.append_data(np.asarray(im))
  if t==800:im.save('videos/g1_fixed_frame.png')
if video:video.close()
arr=np.asarray(records);q=np.asarray(q);desired=np.asarray(desired);errors={m:float(np.sqrt(np.mean((np.asarray(values)-arr[:,4:6])**2))) for m,values in preds.items()}
np.savez_compressed('results/g1_fixed_demo.npz',observations=arr,joint_position=q,joint_target=desired,**{f'prediction_{m}':np.asarray(v) for m,v in preds.items()})
dump('results/g1_fixed_demo.json',dict(fixed_base=True,hardware_used=False,controller='Installed Unitree ImplicitActuator configuration with commanded joint targets, not manufacturer firmware',source_config=str(config_path),source_config_sha256=digest(config_path),asset=cfg.spawn.usd_path,asset_sha256=digest(cfg.spawn.usd_path),body_names=robot.body_names,joint_order=robot.joint_names,actuators=str(cfg.actuators),used_joints=joint_names,used_body=body_names,control_hz=1/DT,seconds=time.time()-start,physical_duration_s=1600*DT,force_rmse_N=errors,joint_tracking_rmse_rad=np.sqrt(np.mean((q-desired)**2,axis=0)).tolist(),max_root_displacement_m=float(np.linalg.norm(robot.data.root_pos_w[0].cpu().numpy()-np.array(cfg.init_state.pos))),claim='Supplementary one-step load prediction on shared observed G1 trajectory; no model-based G1 control comparison',post_primary_test=True))
print('G1_DEMO_COMPLETE',errors,flush=True)
app.close(wait_for_replicator=False,skip_cleanup=True)

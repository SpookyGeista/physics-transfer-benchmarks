"""Generate isolated calibration or frozen test trajectories using existing Isaac Lab."""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from null_physics_transfer.common import CFG, DT, MASS, design, commands, features, history_update, dump
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--splits',nargs='+',default=['train','val','privileged'])
p.add_argument('--count',type=int)
p.add_argument('--steps',type=int)
p.add_argument('--out',default='results/data')
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
app=AppLauncher(a).app
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from pxr import UsdPhysics
from null_physics_transfer.reference import Reference

rows0=design(a.splits[0],a.count)
nmax=max(len(design(s,a.count)) for s in a.splits)
sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT,device=a.device,gravity=(0.,0.,0.)))
for i in range(nmax):
    sim_utils.CuboidCfg(size=(.12,.10,.08),rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True,linear_damping=0.,angular_damping=0.),mass_props=sim_utils.MassPropertiesCfg(mass=MASS),collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)).func(f'/World/env_{i}/Slider',sim_utils.CuboidCfg(size=(.12,.10,.08),rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True,linear_damping=0.,angular_damping=0.),mass_props=sim_utils.MassPropertiesCfg(mass=MASS),collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)),translation=(0.,i*.3,.1))
    joint=UsdPhysics.PrismaticJoint.Define(sim.stage,f'/World/env_{i}/Guide')
    joint.CreateBody1Rel().SetTargets([f'/World/env_{i}/Slider'])
    joint.CreateAxisAttr('X')
    joint.CreateLocalPos0Attr().Set((0.,i*.3,.1))
obj=RigidObject(RigidObjectCfg(prim_path='/World/env_.*/Slider',spawn=None))
sim.reset()
for split in a.splits:
    rows=design(split,a.count); n=len(rows); steps=a.steps or CFG['steps']; cmd=commands(rows,steps)
    slope=np.array([r['slope'] for r in rows]); is_a=np.array([r['mode']=='A' for r in rows]); is_b=np.array([r['mode']=='B' for r in rows])
    for world in CFG['worlds']:
        tic=time.time()
        state=obj.data.default_root_state.clone(); state[:,0]=0; state[:,1]=torch.arange(nmax,device=sim.device)*.3; state[:,2]=.1
        state[:,7:]=0
        obj.write_root_state_to_sim(state); obj.reset()
        ref=Reference(world,n,cmd['d'][:,0]); hist=np.zeros((n,3,2)); hist[:,:,0]=cmd['d'][:,0,None]
        records={k:[] for k in ['state','next_state','features','normal','friction','force','compression','compression_rate']}
        for t in range(steps):
            st=obj.data.root_state_w[:n].cpu().numpy().copy(); x=st[:,0]; v=st[:,7]
            v=np.where(is_a,cmd['prescribed_v'][:,t],v)
            delta=cmd['d'][:,t]+slope*x; rate=cmd['rate'][:,t]+slope*v
            normal,ft=ref.observe_force(delta,rate,v,friction=~is_b)
            if is_a.any():
                # Controlled load friction recalculation with pre-update velocity memory.
                old=ref._old_v
                if world=='sls':
                    mu=.34+.19*np.exp(-(v/.16)**2)+.055*np.abs(old); visc=.35
                else:
                    mu=.27+.21/(1+(v/.12)**2)+.07*np.tanh(cmd['loads'][:,t]/15)+.09*np.abs(old); visc=.55
                normal=np.where(is_a,cmd['loads'][:,t],normal)
                ft=np.where(is_a,-normal*mu*np.tanh(v/.04)-visc*v,ft)
            force=cmd['u'][:,t]-slope*normal+ft
            if is_a.any():
                st_tensor=obj.data.root_state_w.clone()
                st_tensor[:n,7]=torch.as_tensor(v,device=sim.device)
                obj.write_root_velocity_to_sim(st_tensor[:,7:])
            f=torch.zeros((nmax,1,3),device=sim.device); f[:n,0,0]=torch.as_tensor(force,device=sim.device,dtype=torch.float32)
            obj.set_external_force_and_torque(f,torch.zeros_like(f)); obj.write_data_to_sim(); sim.step(render=False); obj.update(DT)
            records['state'].append(np.column_stack([x,v])); records['next_state'].append(obj.data.root_state_w[:n,[0,7]].cpu().numpy().copy())
            records['features'].append(features(delta,rate,v,hist,normal))
            for key,value in [('normal',normal),('friction',ft),('force',force),('compression',delta),('compression_rate',rate)]: records[key].append(value.copy())
            hist=history_update(hist,delta,v)
        out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
        arrays={k:np.stack(v,axis=1) for k,v in records.items()}
        np.savez_compressed(out/f'{world}_{split}.npz',**arrays,**cmd)
        dump(out/f'{world}_{split}.json',dict(world=world,split=split,rows=rows,dt=DT,steps=steps,seconds=time.time()-tic,backend='Isaac Lab / PhysX',device=str(sim.device)))
        print(f'SAVED {world} {split} {n} x {steps}, {time.time()-tic:.2f}s',flush=True)
app.close(wait_for_replicator=False, skip_cleanup=True)

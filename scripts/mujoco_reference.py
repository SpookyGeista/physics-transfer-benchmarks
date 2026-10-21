"""Independent reference implementation in the already installed MuJoCo environment.

--native-friction replaces regularized friction with native joint frictionloss.
No predictor or original Reference class is imported. No robot SDK is imported.
"""
import argparse,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import mujoco
from null_physics_transfer.common import CFG,DT,MASS,design,commands,dump
p=argparse.ArgumentParser();p.add_argument('--out',default='results/mujoco');p.add_argument('--native-friction',action='store_true');p.add_argument('--substeps',type=int,default=1);a=p.parse_args()
rows=design('test');n=len(rows);steps=CFG['steps'];dt=DT/a.substeps;cmd=commands(rows,steps*a.substeps,dt)
xml=f'''<mujoco><option timestep="{dt}" gravity="0 0 0" integrator="Euler" solver="Newton" iterations="100" tolerance="1e-12"/><worldbody><body><joint name="slider" type="slide" axis="1 0 0"/><geom type="box" size=".06 .05 .04" mass="{MASS}" contype="0" conaffinity="0"/></body></worldbody></mujoco>'''
model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
for world in CFG['worlds']:
    tic=time.time();state=np.zeros((n,steps,2));next_state=state.copy();normal=np.zeros((n,steps));friction=normal.copy();compression=normal.copy()
    for e,row in enumerate(rows):
        mujoco.mj_resetData(model,data);s=row['slope'];memory=np.array([cmd['d'][e,0],cmd['d'][e,0],0.])
        for t in range(steps*a.substeps):
            x=float(data.qpos[0]);v=float(data.qvel[0]);raw_delta=cmd['d'][e,t]+s*x;delta=max(raw_delta,0.);rate=(cmd['rate'][e,t]+s*v) if raw_delta>0 else 0.
            if world=='sls':
                N=max(0.,560*delta+360*(delta-memory[0])+1.7e6*delta**3+3.5*rate) if raw_delta>0 else 0.
                mu=.34+.19*np.exp(-(v/.16)**2)+.055*abs(memory[2]);visc=.35 if N>0 else 0.
                memory[0]=np.exp(-dt/.13)*memory[0]+(1-np.exp(-dt/.13))*delta
                memory[2]=np.exp(-dt/.055)*memory[2]+(1-np.exp(-dt/.055))*v
            else:
                N=max(0.,430*delta+7*np.tanh((delta-memory[0])/.012)+190*(delta-memory[1])+7.5e5*delta**3+6*rate) if raw_delta>0 else 0.
                mu=.27+.21/(1+(v/.12)**2)+.07*np.tanh(N/15)+.09*abs(memory[2]);visc=.55 if N>0 else 0.
                memory[0]=np.exp(-dt/.045)*memory[0]+(1-np.exp(-dt/.045))*delta
                memory[1]=np.exp(-dt/.24)*memory[1]+(1-np.exp(-dt/.24))*delta
                memory[2]=np.exp(-dt/.11)*memory[2]+(1-np.exp(-dt/.11))*v
            Ft=-N*mu*np.tanh(v/.04)-visc*v
            if a.native_friction:
                model.dof_frictionloss[0]=N*mu;data.qfrc_applied[0]=cmd['u'][e,t]-s*N-visc*v
            else:
                model.dof_frictionloss[0]=0;data.qfrc_applied[0]=cmd['u'][e,t]-s*N+Ft
            mujoco.mj_step(model,data)
            if a.native_friction:Ft=float(data.qfrc_constraint[0])-visc*v
            # Store states at start and end of each observation interval.
            j=t//a.substeps
            if t%a.substeps==0:
                state[e,j]=[x,v];normal[e,j]=N;friction[e,j]=Ft;compression[e,j]=raw_delta
            if t%a.substeps==a.substeps-1:next_state[e,j]=[data.qpos[0],data.qvel[0]]
    np.savez_compressed(out/f'{world}_test.npz',state=state,next_state=next_state,normal=normal,friction=friction,compression=compression,u=cmd['u'][:,::a.substeps],d=cmd['d'][:,::a.substeps],rate=cmd['rate'][:,::a.substeps])
    dump(out/f'{world}_test.json',dict(world=world,rows=rows,split='test',backend='MuJoCo',version=mujoco.__version__,native_friction=a.native_friction,dt=dt,substeps=a.substeps,seconds=time.time()-tic))
    print('MUJOCO',world,'native',a.native_friction,'substeps',a.substeps,'seconds',time.time()-tic,flush=True)

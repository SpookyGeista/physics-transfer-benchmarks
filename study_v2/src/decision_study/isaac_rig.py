"""Import only after AppLauncher; no DDS, SDK or network robot interface."""
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject,RigidObjectCfg
from pxr import UsdPhysics
from .common import DT,MASS

class IsaacRig:
    def __init__(self,n,device='cpu',dt=DT):
        self.n=n;self.dt=dt
        self.sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt,device=device,gravity=(0.,0.,0.)))
        for i in range(n):
            cfg=sim_utils.CuboidCfg(size=(.12,.10,.08),rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True,linear_damping=0.,angular_damping=0.),mass_props=sim_utils.MassPropertiesCfg(mass=MASS),collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.06,.4,.7)))
            cfg.func(f'/World/env_{i}/Slider',cfg,translation=(0.,i*.3,.1))
            joint=UsdPhysics.PrismaticJoint.Define(self.sim.stage,f'/World/env_{i}/Guide')
            joint.CreateBody1Rel().SetTargets([f'/World/env_{i}/Slider']);joint.CreateAxisAttr('X');joint.CreateLocalPos0Attr().Set((0.,i*.3,.1))
        self.obj=RigidObject(RigidObjectCfg(prim_path='/World/env_.*/Slider',spawn=None))
        self.sim.reset();self.reset()

    def reset(self):
        state=self.obj.data.default_root_state.clone();state[:,0]=0;state[:,1]=torch.arange(self.n,device=self.sim.device)*.3;state[:,2]=.1;state[:,7:]=0
        self.obj.write_root_state_to_sim(state);self.obj.reset()

    def state(self):
        return self.obj.data.root_state_w[:,[0,7]].cpu().numpy().copy()

    def advance(self,force,render=False):
        f=torch.zeros((self.n,1,3),device=self.sim.device);f[:,0,0]=torch.as_tensor(force,device=self.sim.device,dtype=torch.float32)
        self.obj.set_external_force_and_torque(f,torch.zeros_like(f));self.obj.write_data_to_sim();self.sim.step(render=render);self.obj.update(self.dt)
        return self.state()

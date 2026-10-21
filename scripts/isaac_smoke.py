"""Small installed-Isaac-Lab/PhysX force integration test; no robot communications."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device, gravity=(0.0, 0.0, 0.0)))
obj = RigidObject(RigidObjectCfg(prim_path="/World/Slider", spawn=sim_utils.CuboidCfg(
    size=(0.12, 0.10, 0.08), rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0), collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False))))
sim.reset()
force = torch.tensor([[[1., 0., 0.]]], device=sim.device)
for _ in range(100):
    obj.set_external_force_and_torque(force, torch.zeros_like(force))
    obj.write_data_to_sim()
    sim.step(render=False)
    obj.update(sim.get_physics_dt())
result = {"x": obj.data.root_pos_w[0, 0].item(), "v": obj.data.root_lin_vel_w[0, 0].item(),
          "expected_x_continuous": 0.125, "expected_v": 0.5, "device": str(sim.device),
          "torch": torch.__version__, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
result["passed"] = abs(result["v"] - 0.5) < 0.005 and abs(result["x"] - 0.125) < 0.003
Path("results").mkdir(exist_ok=True)
Path("results/isaac_smoke.json").write_text(json.dumps(result, indent=2))
print("SMOKE_RESULT", result, flush=True)
assert result["passed"]
app.close(wait_for_replicator=False, skip_cleanup=True)

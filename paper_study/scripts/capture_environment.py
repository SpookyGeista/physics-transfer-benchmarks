"""Record the already installed simulation/analysis runtime; install nothing."""
from pathlib import Path
import datetime
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    packages={}
    for name in ('numpy','scipy','pandas','torch','numba','matplotlib','isaacsim','isaaclab',
                 'Pillow','imageio','imageio-ffmpeg','pytest'):
        try:packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError:packages[name]=None
    repositories={}
    for path in (Path('/home/loya2/IsaacLab_unitree'),Path('/home/loya2/unitree_rl_lab'),
                 Path('/home/loya2/unitree_mujoco')):
        if path.is_dir():
            result=subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],capture_output=True,text=True,timeout=10)
            repositories[str(path)]={'head':result.stdout.strip() if result.returncode==0 else None}
    payload=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python=sys.executable,python_version=sys.version,platform=platform.platform(),architecture=platform.machine(),
        logical_cpus_visible=os.cpu_count(),packages=packages,repositories=repositories,
        native_device='CPU PhysX; RTX 4090/Vulkan required to initialize installed headless Isaac application',
        robot_scope='Simulation only; no real G1, SDK, ROS or DDS calls in this study',
        installation_policy='Existing environments reused; no simulator or learning package installed',
        document_runtime='/home/loya2/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3')
    path=ROOT/'environment.json'
    with path.open('x') as stream:stream.write(json.dumps(payload,indent=2)+'\n')
    print(path)


if __name__=='__main__':main()

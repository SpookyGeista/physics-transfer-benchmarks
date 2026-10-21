"""Read-only inventory, run in the same host/environment as Isaac Sim."""
import sys,subprocess,json,importlib.metadata as im,platform,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def run(args):
 p=subprocess.run(args,text=True,capture_output=True);return {'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
repos={}
for name in ['IsaacLab','IsaacLab_unitree','unitree_rl_lab','unitree_sim_isaaclab','unitree_mujoco','unitree_sdk2']:
 p=Path('/home/loya2')/name
 repos[name]={'path':str(p),'revision':run(['git','-C',str(p),'rev-parse','HEAD']),'working_tree':run(['git','-C',str(p),'status','--porcelain','-uno'])}
packages={p.metadata['Name']:p.version for p in im.distributions()}
result={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'repos':repos,'packages':packages,'gpu':run(['nvidia-smi']),'memory':run(['free','-h']),'disk':run(['df','-h',str(root)])}
(root/'results/environment.json').write_text(json.dumps(result,indent=2))
(root/'configs/requirements-observed.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(packages.items()))+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['packages','repos']},indent=2))

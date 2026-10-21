import subprocess,time,csv,os
from pathlib import Path
rows=[]
for _ in range(40):
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
 for line in p.stdout.splitlines():
  try:
   pid,mb=[int(x.strip()) for x in line.split(',')]
   cwd=os.readlink(f'/proc/{pid}/cwd');cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
   if cwd=='/home/loya2/physics-transfer':rows.append(dict(timestamp=time.time(),pid=pid,memory_mib=mb,command=cmd))
  except (ValueError,OSError):pass
 time.sleep(.5)
with Path('results/gpu_samples.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['timestamp','pid','memory_mib','command']);w.writeheader();w.writerows(rows)
print('GPU samples',len(rows),flush=True)

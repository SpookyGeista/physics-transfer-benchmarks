"""Frozen local model selection and explicit material wiring; no reference import."""
from pathlib import Path
from online import RecurrentOnline,FeatureOnline,Nominal,Composed,SensorPersistent
from state_id import StateID
ROOT=Path(__file__).resolve().parents[1]/'results/round1/models'

def local(method,world,seed,noise):
    if method=='nominal':return Nominal()
    if method=='recurrent':return RecurrentOnline(ROOT/f'recurrent/r{seed}/{noise}/{world}/model.pt')
    if method in ['old_flex','path_ridge']:return FeatureOnline(ROOT/f'features/{world}/r{seed}/{noise}/{method}/model.json')
    return StateID.load(ROOT/f'state_id/{world}/r{seed}/{noise}/blind_selection/model.json')

def load(method,normal_world,friction_world,seed,noise):
    if method not in ['nominal','recurrent','old_flex','path_ridge','blind_state_id','sensor_state_id','state_reset']:raise ValueError(method)
    n=local(method,normal_world,seed,noise)
    base=n if normal_world==friction_world else Composed(n,local(method,friction_world,seed,noise))
    return SensorPersistent(base) if method=='sensor_state_id' else base

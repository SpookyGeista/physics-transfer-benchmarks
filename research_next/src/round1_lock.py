"""Prospective source and fitted-model locks, with paths relative to project root."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/round1'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify(path):
    path=Path(path);doc=json.loads(path.read_text())
    for relative,expected in doc['files'].items():
        p=ROOT/relative
        if not p.exists() or digest(p)!=expected:raise RuntimeError(f'Locked input changed/missing: {relative}')
    return digest(path)

def verify_code_lock():return verify(OUT/'code_lock.json')

def verify_model_lock():
    code=verify_code_lock();path=OUT/'model_lock.json';doc=json.loads(path.read_text())
    if doc['code_lock_sha256']!=code:raise RuntimeError('Model/source lock mismatch')
    return verify(path)

def verify_test_lock():
    model=verify_model_lock();path=OUT/'test_lock.json';doc=json.loads(path.read_text())
    if doc['model_lock_sha256']!=model:raise RuntimeError('Test/model lock mismatch')
    return verify(path)

def save_lock(path,files,**metadata):
    path=Path(path)
    doc=dict(**metadata,files={os.path.relpath(p.resolve(),ROOT):digest(p) for p in sorted(set(files))})
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(doc,f,indent=2)
    return digest(path)

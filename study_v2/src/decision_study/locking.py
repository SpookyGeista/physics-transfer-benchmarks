import json
from .common import ROOT,digest

def verify_lock():
    path=ROOT/'results/protocol_lock.json';lock=json.loads(path.read_text())
    for name,sha in lock['files'].items():
        assert digest(name)==sha,f'Changed frozen file: {name}'
    return digest(path)

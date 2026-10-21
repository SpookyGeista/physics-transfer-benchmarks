"""Main2 authorization checks without generating any reserved histories."""
from pathlib import Path
import importlib.util
import json
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('main2_native_generator',ROOT/'costed_benchmark/generate_main2.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def config():return json.loads((ROOT/'costed_benchmark/main2_native.json').read_text())


def test_exact_declared_main2_configuration_is_accepted_without_drawing():
    module.validate_config(config())


@pytest.mark.parametrize('field,value',[
    ('preparation_seeds',[53201]),('split','dev'),('dt',.00125),
    ('episodes_per_batch',12),('geometry_ids',[0]),('normal_worlds',['maxwell'])])
def test_unplanned_generation_scope_is_rejected(field,value):
    cfg=config();cfg[field]=value
    with pytest.raises(ValueError):module.validate_config(cfg)


def test_lock_requires_current_config_and_all_sources(tmp_path):
    cfg=config();cfgpath=tmp_path/'cfg.json';cfgpath.write_text(json.dumps(cfg))
    source=tmp_path/'source.py';source.write_text('x=1\n')
    hashes={str(source):module.sha(source)}
    complete={str(p):module.sha(p) for p in module.authorization_source_paths(cfgpath)}
    complete.update(hashes)
    lock=dict(role='paper_study_main2_test_authorization',status='locked',schema_version=1,
              split='test',config_sha256=module.sha(cfgpath),preparation_seeds=cfg['preparation_seeds'],sources_sha256=complete)
    path=tmp_path/'lock.json';path.write_text(json.dumps(lock))
    module.verify_main2_lock(cfg,cfgpath,path,hashes)
    source.write_text('x=2\n')
    with pytest.raises(ValueError,match='source changed'):module.verify_main2_lock(cfg,cfgpath,path,hashes)
    source.write_text('x=1\n');lock['sources_sha256']={};path.write_text(json.dumps(lock))
    with pytest.raises(ValueError,match='omits'):module.verify_main2_lock(cfg,cfgpath,path,hashes)


def authorization(tmp_path):
    cfg=config();cfgpath=tmp_path/'cfg.json';cfgpath.write_text(json.dumps(cfg))
    hashes={str(p):module.sha(p) for p in module.authorization_source_paths(cfgpath)}
    locked=dict(role='paper_study_main2_test_authorization',status='locked',schema_version=1,
        split='test',config_sha256=module.sha(cfgpath),preparation_seeds=cfg['preparation_seeds'],sources_sha256=hashes)
    path=tmp_path/'authorization.json';path.write_text(json.dumps(locked))
    return cfg,cfgpath,path,locked


def test_direct_batch_requires_lock_before_any_reserved_draw(tmp_path,monkeypatch):
    called=[]
    def forbidden(*args,**kwargs):
        called.append(True)
        raise AssertionError('Reserved history would be drawn before authorization')
    monkeypatch.setattr(module,'PreparedPlant',forbidden)
    cfg=config()
    with pytest.raises(ValueError,match='requires prospective'):
        module.run_batch(None,'maxwell','balanced',cfg['probes'][0],'noisy',cfg,[],tmp_path/'unused.npz')
    assert called==[]


@pytest.mark.parametrize('omit_kind',['evaluation','bank3'])
def test_direct_batch_requires_evaluation_and_all_banks_before_draw(tmp_path,monkeypatch,omit_kind):
    cfg,cfgpath,path,locked=authorization(tmp_path)
    if omit_kind=='evaluation':
        remove=next(p for p in locked['sources_sha256'] if p.endswith('/evaluate_main2.py'))
    else:
        remove=next(p for p in locked['sources_sha256'] if '/r61003/' in p and p.endswith('/model.json'))
    locked['sources_sha256'].pop(remove);path.write_text(json.dumps(locked))
    def forbidden(*args,**kwargs):raise AssertionError('Unauthorized history draw')
    monkeypatch.setattr(module,'PreparedPlant',forbidden)
    with pytest.raises(ValueError,match='omits'):
        module.run_batch(None,'maxwell','balanced',cfg['probes'][0],'noisy',cfg,[],tmp_path/'unused.npz',
            config_path=cfgpath,test_lock=path)


def test_in_memory_config_and_out_of_scope_batch_rejected_before_draw(tmp_path,monkeypatch):
    cfg,cfgpath,path,_=authorization(tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError('Out-of-scope history draw')
    monkeypatch.setattr(module,'PreparedPlant',forbidden)
    changed=dict(cfg,max_wall_seconds=cfg['max_wall_seconds']+1)
    with pytest.raises(ValueError,match='In-memory'):
        module.run_batch(None,'maxwell','balanced',cfg['probes'][0],'noisy',changed,[],tmp_path/'unused.npz',
            config_path=cfgpath,test_lock=path)
    with pytest.raises(ValueError,match='outside'):
        module.run_batch(None,'maxwell','balanced',cfg['probes'][0],'noisy',cfg,
            [{'preparation_seed':59999}],tmp_path/'unused.npz',config_path=cfgpath,test_lock=path)

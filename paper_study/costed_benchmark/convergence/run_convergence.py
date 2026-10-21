"""Freeze the declared development resolution diagnostic, then run in order."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE.parent))
import evaluate

PAIRS=('maxwell_balanced','boucwen_wide_scales','maxwell_compliant','boucwen_stiff_high_capacity')
PROBES=('hold_150','gentle_150','strong_300')
PARTICLES=(256,1024,4096)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','run','verify'])
    parser.add_argument('--out',type=Path,default=REPO/'paper_study/results/costed_pi_convergence_v1')
    parser.add_argument('--data',type=Path,default=REPO/'paper_study/results/costed_pi_development')
    args=parser.parse_args()
    out=args.out.resolve()
    if args.stage=='freeze':
        if out.exists() and any(out.iterdir()):raise FileExistsError('Use a fresh convergence directory')
        out.mkdir(parents=True,exist_ok=True)
        sources=evaluate.source_paths()
        evaluate.source_paths=lambda:sources+[Path(__file__).resolve(),HERE/'protocol.md']
        globs=[f'p33001/{pair}/{probe}/noisy.npz' for pair in PAIRS for probe in PROBES]
        hashes={}
        for particles in PARTICLES:
            cfg={**evaluate.DEFAULT_CONFIG,'particles':particles,'calibration_noise':['noisy'],
                'forecast_starts':['filtered_particles'],'max_wall_seconds':7200.,
                'diagnostic_protocol_sha256':evaluate.sha(HERE/'protocol.md')}
            destination=out/f'p{particles}'
            evaluate.freeze(args.data,destination,cfg,batch_globs=globs)
            lock,_=evaluate.verify_lock(destination)
            if len(lock['selected_batches'])!=12:raise ValueError('Expected exactly 12 selected batches')
            hashes[str(destination/'evaluation_lock.json')]=evaluate.sha(destination/'evaluation_lock.json')
        evaluate.write_json(out/'convergence_lock.json',dict(role='All resolution layouts frozen before P256 scoring',
            evaluation_locks_sha256=hashes,protocol_sha256=evaluate.sha(HERE/'protocol.md'),
            script_sha256=evaluate.sha(__file__),particles=PARTICLES,expected_rows_per_P=1296),exclusive=True)
    else:
        lock=json.loads((out/'convergence_lock.json').read_text())
        evaluate.verify_hashes(lock['evaluation_locks_sha256'])
        if evaluate.sha(HERE/'protocol.md')!=lock['protocol_sha256'] or evaluate.sha(__file__)!=lock['script_sha256']:
            raise ValueError('Convergence source/protocol changed')
        for particles in PARTICLES:
            destination=out/f'p{particles}'
            evaluate.verify_lock(destination)
            if args.stage=='run':evaluate.run(destination)
            evaluate.verify_lock(destination)
        evaluate.verify_hashes(lock['evaluation_locks_sha256'])


if __name__=='__main__':main()

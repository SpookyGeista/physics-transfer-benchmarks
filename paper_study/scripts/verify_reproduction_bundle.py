"""Stream every archive member and verify its bytes without extracting files."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,tarfile,time


def digest(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(4*1024*1024),b''):
        h.update(block)
    return h.hexdigest()


def safe_relative(name,basename=False):
    path=PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or str(path)!=name or '\\' in name:
        raise ValueError('Unsafe archive path: '+name)
    if basename and len(path.parts)!=1:
        raise ValueError('Archive must have a basename only: '+name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    if a.out.exists():
        raise ValueError('Verification output must be fresh')
    source=a.bundle/'MANIFEST.json'
    manifest=json.loads(source.read_text())
    start=time.perf_counter();results=[]
    for record in manifest['archives']:
        safe_relative(record['file'],basename=True)
        path=a.bundle/record['file']
        if path.stat().st_size!=record['size']:
            raise ValueError('Archive size changed: '+str(path))
        with path.open('rb') as stream:
            actual=digest(stream)
        if actual!=record['sha256']:
            raise ValueError('Archive checksum failed: '+str(path))
        expected={x['path']:x for x in record['members']}
        for name in expected:
            safe_relative(name)
        if len(expected)!=len(record['members']):
            raise ValueError('Duplicate manifest member')
        seen=set();total=0
        with tarfile.open(path,mode='r|*') as archive:
            for member in archive:
                safe_relative(member.name)
                if not member.isfile() or member.name in seen or member.name not in expected:
                    raise ValueError('Unexpected, duplicate or non-file archive member: '+member.name)
                wanted=expected[member.name]
                if member.size!=wanted['size']:
                    raise ValueError('Wrong member size: '+member.name)
                with archive.extractfile(member) as stream:
                    if digest(stream)!=wanted['sha256']:
                        raise ValueError('Member checksum failed: '+member.name)
                seen.add(member.name);total+=member.size
        if seen!=set(expected):
            raise ValueError('Missing archive members')
        results.append(dict(archive=path.name,sha256=actual,members=len(seen),verified_member_bytes=total))
    output=dict(status='complete',manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        seconds=time.perf_counter()-start,archives=results,
        scope='All complete archive bytes and all member bytes independently streamed; no extraction',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as stream:
        json.dump(output,stream,indent=2);stream.write('\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()

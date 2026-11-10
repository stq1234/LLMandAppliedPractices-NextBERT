"""Recover official hash-verified wheels from the current user's pip cache offline."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--destination', required=True)
args = p.parse_args()
destination = Path(args.destination).resolve()
destination.relative_to(Path.home() / 'LLM')
destination.mkdir(parents=True, exist_ok=True)
expected = json.loads(Path(args.manifest).read_text())
by_hash = {x['sha256']: x['filename'] for x in expected}
wanted = {x['filename'] for x in expected}
for source in (Path.home() / '.cache' / 'pip').rglob('*.body'):
    if not zipfile.is_zipfile(source):
        continue
    with zipfile.ZipFile(source) as z:
        wheels = [n for n in z.namelist() if n.endswith('.dist-info/WHEEL')]
        if not wheels:
            continue
        dist = wheels[0].split('/')[0].removesuffix('.dist-info')
        if not any(name.startswith(dist + '-') for name in wanted):
            continue
    with source.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    filename = by_hash.get(digest)
    if filename:
        target = destination / filename
        if target.exists():
            with target.open('rb') as f:
                if hashlib.file_digest(f, 'sha256').hexdigest() != digest:
                    raise RuntimeError(f'Existing target mismatch: {filename}')
        else:
            shutil.copyfile(source, target)
        print('verified', filename, flush=True)
missing = sorted(wanted - {x.name for x in destination.glob('*.whl')})
print(json.dumps({'missing': missing, 'verified_count': len(wanted) - len(missing)}))

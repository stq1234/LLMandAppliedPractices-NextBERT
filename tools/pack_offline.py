"""Build a byte-preserving archive for the shared verified upload transport."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('paths', nargs='+')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = (root / args.output).resolve()
    target.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for name in args.paths:
        source = (root / name).resolve()
        source.relative_to(root)
        if not source.exists():
            raise FileNotFoundError(source)
        for f in sorted(source.rglob('*')) if source.is_dir() else [source]:
            if f.is_file() and not f.is_symlink() and not any(
                part in {'.cache', '.download-parts', '__pycache__', '.pytest_cache', '.git'} for part in f.parts
            ):
                if f.resolve() == target:
                    raise ValueError('Archive cannot include itself')
                files.append(f)
    mode = 'w:gz' if target.name.endswith(('.tar.gz', '.tgz')) else 'w'
    with tarfile.open(target, mode) as archive:
        for f in sorted(set(files)):
            info = archive.gettarinfo(str(f), arcname=f.relative_to(root).as_posix())
            info.mode, info.uid, info.gid = 0o644, 0, 0
            info.uname = info.gname = ''
            with f.open('rb') as stream:
                archive.addfile(info, stream)
    with target.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest = {'archive': target.name, 'bytes': target.stat().st_size,
                'sha256': digest, 'members': [f.relative_to(root).as_posix() for f in sorted(set(files))]}
    target.with_suffix(target.suffix + '.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'members'}))


if __name__ == '__main__':
    main()

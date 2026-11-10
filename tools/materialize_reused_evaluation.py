"""Reuse identical pinned evaluation assets, verifying hashes before linking."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    profile = ROOT / 'data/prepared/full_67m'
    meta = json.loads((profile / 'manifest_retrieval.json').read_text(encoding='utf-8'))
    names = ('retrieval_corpus', 'retrieval_queries_dev', 'retrieval_queries_test',
             'reranking_dev', 'reranking_test')
    completed = []
    for name in names:
        expected = meta['artifacts'][name]['sha256']
        target = profile / f'{name}.jsonl'
        source = ROOT / 'data/prepared/full' / f'{name}.jsonl'
        target.resolve().relative_to(profile.resolve())
        if target.exists():
            if digest(target) != expected:
                raise RuntimeError(f'Existing target has unexpected content: {target}')
            completed.append({'name': name, 'method': 'already_verified'})
            continue
        if digest(source) != expected:
            raise RuntimeError(f'Reuse source differs from new profile manifest: {source}')
        try:
            os.link(source, target)
            method = 'hardlink'
        except OSError:
            shutil.copyfile(source, target)
            method = 'copy'
        if digest(target) != expected:
            raise RuntimeError(f'Reused target verification failed: {target}')
        completed.append({'name': name, 'method': method})
    print(json.dumps({'verified_reused_assets': completed}))


if __name__ == '__main__':
    main()

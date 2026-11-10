"""Archive moderate prepared files; record hashes/rebuild paths for large files."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-file-mib', type=int, default=96)
    args = parser.parse_args()
    output = ROOT / 'data/archives'
    output.mkdir(exist_ok=True)
    inventory = {'archive_root': 'repository root', 'large_file_rule_mib': args.max_file_mib,
                 'rebuild_command': 'python src/prepare_data.py --profile full_67m --plan configs/experiment_plan.json',
                 'legacy_rebuild_command': 'python src/prepare_data.py --profile full --plan configs/calibration_small_plan.json',
                 'included': [], 'excluded_large_files': []}
    commands = []
    for profile in ('pilot', 'full', 'full_67m'):
        source = ROOT / 'data/prepared' / ('full' if profile == 'full' else '')
        if profile == 'full_67m':
            source = ROOT / 'data/prepared/full_67m'
        if not source.is_dir():
            continue
        for task in ('wiki', 'classification', 'qa', 'retrieval'):
            prefixes = {'wiki': ('wiki_',), 'classification': ('classification_',),
                        'qa': ('qa_',), 'retrieval': ('retrieval_', 'reranking_')}[task]
            selected = []
            for path in sorted(source.iterdir()):
                if not path.is_file() or not (path.name.startswith(prefixes) or path.name == f'manifest_{task}.json'):
                    continue
                rel = path.relative_to(ROOT).as_posix()
                if path.stat().st_size >= args.max_file_mib * 1024**2:
                    with path.open('rb') as stream:
                        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                    inventory['excluded_large_files'].append({'path': rel, 'bytes': path.stat().st_size, 'sha256': digest})
                else:
                    selected.append(rel)
            if selected:
                archive = output / f'{profile}_{task}.tar.gz'
                commands.append((archive, [sys.executable, str(ROOT / 'tools/pack_offline.py'), '--output', str(archive), *selected]))
                inventory['included'].append(str(archive.relative_to(ROOT)).replace('\\', '/'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(archive, pool.submit(subprocess.run, command, check=True)) for archive, command in commands]
        for archive, future in futures:
            future.result()
            if archive.stat().st_size >= 96 * 1024**2:
                raise RuntimeError(f'Archive is too large for normal Git: {archive.name}; split its input groups')
    (ROOT / 'data/archive_manifest.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    print(json.dumps({'archives': len(inventory['included']), 'excluded_large_files': len(inventory['excluded_large_files'])}))


if __name__ == '__main__':
    main()

"""Install task archives within this checkout, refusing unexpected overwrites."""
import argparse
import hashlib
from pathlib import Path
import tarfile


def digest_stream(stream):
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--expected-bundle', type=Path, action='append', default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = {}
    for prior_bundle in args.expected_bundle:
        with tarfile.open(prior_bundle) as prior:
            for member in prior.getmembers():
                if member.isfile():
                    expected[member.name] = digest_stream(prior.extractfile(member))
    with tarfile.open(args.bundle) as archive:
        members = archive.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            target.relative_to(root)
            if not member.isfile() or target.is_symlink():
                raise ValueError(f'Unsupported archive member: {member.name}')
            if target.exists():
                with target.open('rb') as stream:
                    current = digest_stream(stream)
                incoming = digest_stream(archive.extractfile(member))
                if current != incoming and current != expected.get(member.name):
                    raise RuntimeError(f'Unexpected remote content: {member.name}')
        for member in members:
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            with archive.extractfile(member) as src, target.open('wb') as dst:
                import shutil
                shutil.copyfileobj(src, dst)
            target.chmod(0o644)
    print(f'Installed and checked {len(members)} files from {args.bundle.name}')


if __name__ == '__main__':
    main()

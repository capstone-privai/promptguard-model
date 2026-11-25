"""What fills the Drive: sizes of promptguard-runs entries and of every large file in My Drive, and which large files
are byte-identical to a Git LFS object of this repository (so a copy already lives on GitHub).

  python scripts/drive_inventory.py [--drive /content/drive/MyDrive] [--min-mb 50]

Run on Colab with Drive mounted, from a clone made with GIT_LFS_SKIP_SMUDGE=1 (models/ then holds LFS pointer files,
which give each object's sha256 and size). A Drive file is hashed only when its size equals an LFS object's size.
Read-only: prints names, sizes and matches; deletes nothing and reads no file contents beyond hashing.
"""
import argparse
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def lfs_objects():
    """{(sha256, size): repo path} from the LFS pointer files under models/."""
    objects = {}
    for path in (ROOT / 'models').rglob('*'):
        if not path.is_file() or path.stat().st_size > 1024:
            continue
        text = path.read_text(errors='ignore')
        if not text.startswith('version https://git-lfs'):
            continue
        fields = dict(line.split(' ', 1) for line in text.splitlines() if ' ' in line)
        objects[(fields['oid'].removeprefix('sha256:'), int(fields['size']))] = str(path.relative_to(ROOT))
    return objects


def size_of(path):
    if path.is_file():
        return path.stat().st_size
    total = 0
    for directory, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(directory, name))
            except OSError:
                pass
    return total


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--drive', default='/content/drive/MyDrive')
    p.add_argument('--min-mb', type=float, default=50)
    args = p.parse_args()
    drive, runs = Path(args.drive), Path(args.drive) / 'promptguard-runs'
    objects = lfs_objects()
    sizes = {size for _, size in objects}
    print(f'LFS objects in this checkout: {len(objects)}')
    print('\n== promptguard-runs entries (GB)')
    entries = sorted(((size_of(e), e.name) for e in runs.iterdir()), reverse=True)
    for size, name in entries:
        print(f'{size / 2**30:7.2f}  {name}')
    print(f'{sum(s for s, _ in entries) / 2**30:7.2f}  (total)')
    print(f'\n== files over {args.min_mb:g} MB in My Drive (MB, path, copy on GitHub LFS)')
    large = []
    for directory, _, files in os.walk(drive):
        for name in files:
            path = Path(directory) / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size >= args.min_mb * 2**20:
                large.append((size, path))
    for size, path in sorted(large, reverse=True):
        match = objects.get((sha256(path), size)) if size in sizes else None
        print(f'{size / 2**20:8.0f}  {path.relative_to(drive)}  {"= " + match if match else ""}', flush=True)
    print(f'{sum(s for s, _ in large) / 2**30:8.2f} GB in these files')


if __name__ == '__main__':
    main()

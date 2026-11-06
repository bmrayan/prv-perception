"""Install the abstract 2D release from a local dataset directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'hf-perception-not-reasoning')
    args = parser.parse_args()
    source = args.source.resolve()
    probes = json.loads((source / 'probes.json').read_text(encoding='utf-8'))
    checksums = json.loads((source / 'checksums.json').read_text(encoding='utf-8'))
    for rel, expected in checksums.items():
        path = (source / rel).resolve()
        if not path.is_relative_to(source):
            raise ValueError(f'Path outside dataset: {rel}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {rel}')
    if len(probes) != 1000 or len({p['id'] for p in probes}) != 1000:
        raise ValueError('Expected 1,000 unique probes')
    target = ROOT / 'data' / 'abstract2d'
    target.mkdir(parents=True, exist_ok=True)
    for probe in probes:
        image = source / probe['image_path']
        shutil.copy2(image, target / image.name)
        probe['image_path'] = f'data/abstract2d/{image.name}'
    (target / 'probes.json').write_text(json.dumps(probes, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Installed {len(probes)} verified probes in {target}')


if __name__ == '__main__':
    main()

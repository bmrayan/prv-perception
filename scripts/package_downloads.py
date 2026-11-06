"""Build local website download archives without uploading."""
import argparse
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT.parent/'hf-perception-not-reasoning')
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    subprocess.run([sys.executable, str(dataset/'validate_dataset.py')], check=True)
    target = ROOT/'website/assets'
    with zipfile.ZipFile(target/'abstract2d.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dataset.rglob('*')):
            if path.is_file() and not any(part in path.parts for part in ('.git','__pycache__','.venv','.cache')):
                archive.write(path, path.relative_to(dataset).as_posix())
    with zipfile.ZipFile(target/'source-code.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob('*')):
            rel = path.relative_to(ROOT)
            if not path.is_file() or any(part in rel.parts for part in ('.git','__pycache__','.venv')):
                continue
            if rel.as_posix().startswith(('results/generated/','results/reproduced/','data/abstract2d/')):
                continue
            if path.suffix == '.zip' or path.with_suffix(path.suffix+'.gz').exists():
                continue
            archive.write(path, rel.as_posix())
    print(f'Local archives written to {target}')


if __name__ == '__main__':
    main()

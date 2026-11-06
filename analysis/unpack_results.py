"""Expand saved results for the original experiment-specific analysis tools."""
import gzip
from pathlib import Path
import shutil

from reproduce import ROOT


def main():
    count = 0
    for path in (ROOT / 'results').rglob('*.gz'):
        target = path.with_suffix('')
        if target.exists():
            continue
        with gzip.open(path, 'rb') as src, target.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        count += 1
    print(f'Expanded {count} saved result files. Compressed originals are unchanged.')


if __name__ == '__main__':
    main()

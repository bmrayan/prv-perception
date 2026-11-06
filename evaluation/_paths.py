"""Locations outside the code checkout for datasets, benchmarks, and run artifacts."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('PERCEPTION_DATA_DIR', REPO_ROOT.parent / 'hf-perception-not-reasoning')).expanduser().resolve()
BENCHMARK_ROOT = Path(os.environ.get('PERCEPTION_BENCHMARK_DIR', REPO_ROOT.parent / 'benchmark-data')).expanduser().resolve()

def resolve_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'data':
        return BENCHMARK_ROOT.joinpath(*path.parts[1:])
    return ROOT / path

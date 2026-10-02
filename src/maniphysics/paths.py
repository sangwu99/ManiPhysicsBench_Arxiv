import os
from pathlib import Path

def release_root():
    return Path(__file__).resolve().parents[2]

def external_root():
    return Path(os.environ.get('MPB_EXTERNAL_ROOT', release_root() / 'runtime/external'))

def runtime_root():
    return Path(os.environ.get('MPB_RUNTIME_ROOT', release_root() / 'runtime'))

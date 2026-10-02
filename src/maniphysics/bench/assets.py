import json
from pathlib import Path
from maniphysics.evaluation.metadata import fingerprint
from maniphysics.paths import release_root
from .obj_parts import split_obj_parts

def manifest():
    return json.loads((release_root() / 'bench/assets/manifest.json').read_text())

def checked_path(record):
    path = release_root() / record['path']
    if fingerprint(path)['sha256'] != record['sha256']:
        raise ValueError(f"Rollout asset hash mismatch: {record['path']}")
    return path

def model_files(name):
    record = manifest()['objects'][name]
    return {key: checked_path(value) for (key, value) in record['files'].items()}

def properties(name):
    data = manifest()
    registry = json.loads(checked_path(data['registry']).read_text())
    return {**data['defaults'], **registry[name]}

def libero_collision(name, directory):
    data = manifest()
    files = model_files(name)
    target = Path(directory) / name
    parts = split_obj_parts(files['collision.obj'], target)
    return dict(parts=[str(target / name) for name in parts], transform=data['objects'][name]['libero'], properties=properties(name), contact=data['libero_contact'])

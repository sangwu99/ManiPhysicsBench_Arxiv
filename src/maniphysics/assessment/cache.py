import hashlib
import json
from importlib.metadata import version
from pathlib import Path


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def signature(grasp, profile, runtime):
    from maniphysics.zoo.catalog import geometry_path, card
    source = Path(__file__).parent
    code = {str(p.relative_to(source)): file_hash(p) for p in sorted(source.rglob('*.py'))}
    mesh = geometry_path(profile['spec'].get('geometry_id', grasp['object_id']))
    return dict(object_id=grasp['object_id'], points_m=grasp['points_m'], axis=grasp['axis'],
                profile=profile, mesh_sha256=file_hash(mesh), card=card(grasp['object_id']),
                code=code, runtime=runtime,
                dependencies={name: version(name) for name in ('numpy', 'scipy', 'trimesh', 'gmsh')})


def donor_key(provenance, grasp, coupled):
    evidence = {k: v for k, v in provenance.items() if k not in ('points_m', 'axis')}
    return digest(dict(evidence=evidence, kp_N_m=grasp['kp_N_m'],
                       cap=grasp.get('actuator_limit_N'), coupled=coupled))


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def seal(directory):
    directory = Path(directory)
    names = ('case.json', 'geometry.npz', 'result.json', 'input_audit.json',
             'curve.json', 'inertia_audit.json', 'execution.json')
    selected = [p for p in directory.rglob('*') if p.is_file() and
                (p.name in names or p.suffix in ('.inp', '.feb', '.rad'))]
    hashes = {str(p.relative_to(directory)): file_hash(p) for p in selected}
    save_json(directory / 'artifacts.json', hashes)


def verify(directory):
    directory = Path(directory)
    hashes = json.loads((directory / 'artifacts.json').read_text())
    for name, expected in hashes.items():
        if file_hash(directory / name) != expected:
            raise ValueError(f'Cached artifact changed: {name}')

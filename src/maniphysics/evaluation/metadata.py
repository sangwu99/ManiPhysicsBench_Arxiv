import hashlib
import json
import math
from pathlib import Path
import re
from maniphysics.zoo.catalog import object_id

def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda : stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(file=path.name, sha256=digest.hexdigest())

def success_value(record):
    values = [record[k] for k in ('task_success', 'success') if record.get(k) is not None]
    if any((type(v) is not bool for v in values)):
        raise ValueError('Episode success must be a boolean')
    if len(set(values)) > 1:
        raise ValueError('Conflicting task_success and success')
    return values[0] if values else None

def episode_index(record):
    values = [record[k] for k in ('episode', 'ep') if record.get(k) is not None]
    if not values or any((type(v) is not int or v < 0 for v in values)):
        raise ValueError('An explicit nonnegative episode index is required')
    if len(set(values)) != 1:
        raise ValueError('Conflicting episode indices')
    return values[0]

def summary_success(path, task, episode):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and 'results' in data:
        data = data['results']
    if isinstance(data, dict) and task in data:
        data = data[task]
    elif isinstance(data, dict) and data.get('env') != task:
        raise ValueError(f'Summary does not identify task {task}')
    if isinstance(data, dict):
        keys = [key for key in ('episodes', 'per_episode', 'rows') if key in data]
        if len(keys) != 1:
            raise ValueError('Summary must contain one episode list')
        data = data[keys[0]]
    if not isinstance(data, list):
        raise ValueError('Summary has no episode list')
    rows = [row for row in data if episode_index(row) == episode]
    if len(rows) != 1:
        raise ValueError(f'Expected one summary row for episode {episode}, found {len(rows)}')
    if rows[0].get('error'):
        raise ValueError('The selected summary episode contains an execution error')
    result = success_value(rows[0])
    if result is None:
        raise ValueError('Summary episode has no boolean success')
    return result

def controller_value(metadata, key, explicit):
    logged = metadata.get(key)
    for value in (logged, explicit):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
            raise ValueError(f'{key} must be a finite positive scalar')
    if logged is not None and explicit is not None and (not math.isclose(logged, explicit)):
        raise ValueError(f'{key} conflicts with the recorded controller')
    if logged is None and explicit is None:
        raise ValueError(f'Missing {key}; provide the actual controller setting explicitly')
    return dict(value=float(logged if logged is not None else explicit), source='trajectory_metadata' if logged is not None else 'explicit')

def resolve(trajectory, name, *, summary=None, task=None, summary_episode=None, kp_N_m=None, actuator_limit_N=None):
    trajectory = Path(trajectory)
    sidecar = trajectory.with_suffix('.json')
    metadata = json.loads(sidecar.read_text())
    if metadata.get('execution_error'):
        raise ValueError('Trajectory records an execution error')
    mid = object_id(name)
    identities = [metadata[k] for k in ('object_id', 'model_id') if metadata.get(k) is not None]
    if not identities or any((object_id(value) != mid for value in identities)):
        raise ValueError('Requested object does not match the recorded object identity')
    episode = episode_index(metadata)
    match = re.search('_ep(\\d+)_traj$', trajectory.stem)
    if match and int(match.group(1)) != episode:
        raise ValueError('Filename episode index disagrees with trajectory metadata')
    success = success_value(metadata)
    evidence = dict(trajectory=fingerprint(trajectory), metadata=fingerprint(sidecar), episode=episode, object_id=mid)
    if summary is not None:
        if task is None or summary_episode is None:
            raise ValueError('Summary joins require explicit task and summary_episode')
        if type(summary_episode) is not int or summary_episode < 0:
            raise ValueError('summary_episode must be a nonnegative integer')
        recorded_task = metadata.get('task_id', metadata.get('env'))
        if recorded_task is not None and recorded_task != task:
            raise ValueError('Summary task disagrees with trajectory metadata')
        from maniphysics.bench.libero.tasks import TASK_BY_SUFFIX
        if trajectory.parent.name in TASK_BY_SUFFIX and trajectory.parent.name != task:
            raise ValueError('Summary task disagrees with the trajectory task directory')
        joined = summary_success(summary, task, summary_episode)
        if success is not None and joined != success:
            raise ValueError('Summary and trajectory success disagree')
        success = joined
        evidence.update(summary=fingerprint(summary), summary_task=task, summary_episode=summary_episode)
    elif task is not None or summary_episode is not None:
        raise ValueError('task and summary_episode require a summary file')
    if success is None:
        raise ValueError('Missing episode success; supply its task summary and episode mapping')
    controller = dict(kp_N_m=controller_value(metadata, 'gripper_kp', kp_N_m), actuator_limit_N=controller_value(metadata, 'gripper_force_limit', actuator_limit_N))
    evidence['controller'] = controller
    return dict(task_success=success, object_id=mid, episode=episode, kp_N_m=controller['kp_N_m']['value'], actuator_limit_N=controller['actuator_limit_N']['value'], provenance=evidence)

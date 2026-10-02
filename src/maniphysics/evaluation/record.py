import json
from pathlib import Path

def finalize_episode(env, success, episode, task, error=None):
    if hasattr(env, 'unwrapped'):
        e = env.unwrapped
    else:
        e = env.env
    if not e._mp_log_dir:
        raise RuntimeError('A trajectory output directory is required')
    path = Path(e._mp_log_dir) / f'{e._mp_env_tag}_ep{e._mp_ep_idx:04d}_traj.npz'
    if hasattr(e, '_mp_pending'):
        e._mp_pending['success'] = bool(success) if error is None else None
        e._mp_dump()
    else:
        e._mp_dump(success=bool(success) if error is None else None)
    sidecar = path.with_suffix('.json')
    meta = json.loads(sidecar.read_text())
    meta.update(task=task, task_id=task, evaluator_episode=int(episode), execution_error=error)
    sidecar.write_text(json.dumps(meta, indent=2) + '\n')
    manifest = path.parent / 'episodes.jsonl'
    with manifest.open('a') as stream:
        stream.write(json.dumps(dict(task=task, episode=int(episode), trajectory=path.name, task_success=meta['success'], error=error)) + '\n')
    if error is not None:
        raise RuntimeError(error)
    return path

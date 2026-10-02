import json
from pathlib import Path
import numpy as np
from maniphysics.zoo.catalog import object_id
from .objects import profile
from .cache import signature, digest, save_json, seal, verify, donor_key
from .servo import threshold, classify
from .neighbors import estimate


def validate(grasp):
    result = dict(grasp, object_id=object_id(grasp['object_id']))
    points, axis = np.asarray(result['points_m'], float), np.asarray(result['axis'], float)
    if points.shape != (2, 3) or axis.shape != (3,) or not np.isfinite(points).all() or not np.isfinite(axis).all():
        raise ValueError('Two finite contact points and a finite closing axis are required')
    chord = points[1] - points[0]
    if np.linalg.norm(chord) == 0 or np.linalg.norm(axis) == 0:
        raise ValueError('Degenerate grasp')
    chord /= np.linalg.norm(chord)
    axis /= np.linalg.norm(axis)
    if not np.isclose(abs(chord @ axis), 1., rtol=0, atol=1e-8):
        raise ValueError('Closing axis must follow the contact chord')
    if result['coordinate_frame'] != 'asset_bbox_centered':
        raise ValueError('Convert logged contact points to the asset frame first')
    if result['contact_duration_s'] < .2:
        raise ValueError('Grasp duration must be at least 0.2 seconds')
    if not np.isfinite(result['observed_force_N']) or result['observed_force_N'] < 0:
        raise ValueError('Invalid observed force')
    result['axis'] = axis.tolist()
    return result


def prepare(grasp, output, runtime):
    from .core import reference_contact
    grasp = validate(grasp)
    settings = profile(grasp['object_id'])
    runtime.bind(settings['backend'])
    provenance = signature(grasp, settings, runtime.fingerprint(settings['backend']))
    key = digest(provenance)
    directory = Path(output).resolve() / key
    if directory.exists():
        if json.loads((directory / 'provenance.json').read_text()) != provenance:
            raise ValueError('Cache provenance mismatch')
        if (directory / 'curve.json').exists():
            verify(directory)
        return grasp, settings, directory, key
    directory.mkdir(parents=True)
    save_json(directory / 'provenance.json', provenance)
    save_json(directory / 'grasp.json', grasp)
    points = np.asarray(grasp['points_m'])
    axis = np.asarray(grasp['axis'])
    if (points[1] - points[0]) @ axis < 0:
        axis = -axis
    reference_contact.prepare(directory / 'case', grasp['object_id'], points, np.array([axis, -axis]),
                              settings['spec'])
    return grasp, settings, directory, key


def _assess(grasp, output, runtime, *, coupled=True, donors=(), prepare_only=False):
    from .solvers import backends
    grasp, settings, directory, key = prepare(grasp, output, runtime)
    if prepare_only:
        return dict(id=key, object_id=grasp['object_id'], backend=settings['backend'], status='prepared')
    if (directory / 'curve.json').exists():
        curve = json.loads((directory / 'curve.json').read_text())
    else:
        if (directory / 'execution_started.json').exists():
            raise RuntimeError('An incomplete execution exists; preserve it and choose a new output directory')
        save_json(directory / 'execution_started.json', dict(backend=settings['backend']))
        curve = getattr(backends, settings['backend'])(directory / 'case', directory, settings, runtime)
        save_json(directory / 'curve.json', curve)
        seal(directory)
    bounds = threshold(curve['rows'], kp_N_m=grasp['kp_N_m'],
                       actuator_limit_N=grasp.get('actuator_limit_N'), coupled=coupled)
    raw = threshold(curve['rows'], kp_N_m=grasp['kp_N_m'], coupled=False)
    compatibility = donor_key(json.loads((directory / 'provenance.json').read_text()), grasp, coupled)
    result = dict(id=key, object_id=grasp['object_id'], points_m=grasp['points_m'],
                  observed_force_N=grasp['observed_force_N'], threshold=bounds,
                  raw_threshold=raw, backend=settings['backend'], origin='direct',
                  compatibility=compatibility, task_success=grasp.get('task_success'))
    verdict = classify(bounds, grasp['observed_force_N'])
    execution = curve.get('execution') or {}
    failed = execution.get('returncode', 0) != 0 and execution.get('reason') != 'intentional_stop_after_verified_first_damage'
    if verdict['status'] == 'unknown' and donors and (bounds['kind'] == 'failed' or
            failed and bounds['kind'] == 'right_censored'):
        inferred = estimate(result, donors)
        if inferred['kind'] == 'neighbor_median':
            inferred.update({k: bounds[k] for k in ('force_convention', 'coupled', 'kp_N_m', 'actuator_limit_N')})
            result.update(threshold=inferred, origin='estimated', failed_direct_threshold=bounds)
            verdict = classify(inferred, grasp['observed_force_N'])
    result.update(verdict=verdict, raw_verdict=classify(raw, grasp['observed_force_N']),
                  curve_states=len(curve['rows']), solver_execution=execution,
                  requested_path_completed=bool(curve.get('completed')),
                  prefix_reason=curve.get('prefix_reason') or curve.get('reason'))
    scoring_key = digest(dict(observed=grasp['observed_force_N'], kp=grasp['kp_N_m'],
                             cap=grasp.get('actuator_limit_N'), coupled=coupled, threshold=result['threshold']))
    save_json(directory / f'verdict_{scoring_key[:16]}.json', result)
    return result


def assess(grasp, output, runtime, *, coupled=True, donors=(), prepare_only=False):
    import fcntl
    locks = Path(output).resolve() / 'locks'
    locks.mkdir(parents=True, exist_ok=True)
    key = digest(dict(object=object_id(grasp['object_id']), points=grasp['points_m'], axis=grasp['axis']))
    with (locks / key).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        return _assess(grasp, output, runtime, coupled=coupled, donors=donors, prepare_only=prepare_only)


def aggregate(results, *, task_success, coverage_complete=True):
    statuses = [x['verdict']['status'] for x in results]
    preservation = ('damaged' if 'damaged' in statuses else
                    'unknown' if not statuses or not coverage_complete or 'unknown' in statuses else 'safe')
    return dict(task_success=bool(task_success), preservation=preservation,
                safe_success=bool(task_success and preservation == 'safe'),
                assessment_complete=preservation != 'unknown', grasps=len(results))

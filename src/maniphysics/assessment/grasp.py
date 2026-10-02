import numpy as np
from .core.events import extract as extract_events
from .core.grasp import log_to_mesh_frame
from maniphysics.zoo.catalog import object_id
from maniphysics.evaluation.metadata import resolve


CONTACT_DRIFT_M = {'chicken-egg': .0005115, 'cherry-tomato-w30l40mm': .001,
                   'walnut-in-shell': .001, 'boiled-potato': .001, 'strawberry': .001,
                   'plum': .001, 'paper-cup-2oz': .00064125, 'paper-cup-takeaway': .000745,
                   'portion-cup-1oz-pp': .00060525, 'yakult-bottle': .000943}


def extract(trajectory, name, *, kp_N_m=None, actuator_limit_N=None,
            summary=None, task=None, summary_episode=None):
    mid = object_id(name)
    record = resolve(trajectory, mid, kp_N_m=kp_N_m, actuator_limit_N=actuator_limit_N,
                     summary=summary, task=task, summary_episode=summary_episode)
    data = extract_events(trajectory, sustain_s=.2, max_drift_m=CONTACT_DRIFT_M[mid])
    grasps = []
    for event in data['events']:
        frame = event['geometry_frame']
        points = np.array([log_to_mesh_frame(mid, event[key], frame=frame) for key in ('pL', 'pR')])
        axis = points[1] - points[0]
        axis /= np.linalg.norm(axis)
        grasps.append(dict(id=str(event['i']), object_id=mid, points_m=points.tolist(), axis=axis.tolist(),
                           observed_force_N=event['fL'], contact_duration_s=event['contact_s'],
                           coordinate_frame='asset_bbox_centered', kp_N_m=record['kp_N_m'],
                           actuator_limit_N=record['actuator_limit_N']))
    return dict(grasps=grasps, coverage_reasons=data['coverage_reasons'],
                task_success=record['task_success'], object_id=mid, episode=record['episode'],
                provenance=record['provenance'],
                sampling=data['sampling'], configuration=data.get('configuration'),
                excluded=data.get('excluded'),
                force_component=data['metadata'].get('force_component'))

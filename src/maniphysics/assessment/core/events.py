import json
from pathlib import Path
import numpy as np
SUSTAIN_S = 0.2

def max_sustained(values, n):
    v = np.asarray(values, float)
    if n < 1:
        raise ValueError('window must contain at least one sample')
    if len(v) < n:
        return (0.0, None)
    windows = np.lib.stride_tricks.sliding_window_view(v, n)
    mins = windows.min(axis=1)
    j = int(np.argmax(mins))
    return (float(mins[j]), j + int(np.argmin(windows[j])))

def extract(path, *, sim_freq=None, sustain_s=SUSTAIN_S, max_drift_m=0.001, max_angle_deg=5.0):
    path = Path(path)
    side = path.with_suffix('.json')
    meta = json.loads(side.read_text()) if side.exists() else {}
    frame = {k: meta.get(k) for k in ('platform', 'robot_uid', 'mesh_in_object_quat_wxyz')}
    hz = float(meta.get('sim_freq', sim_freq if sim_freq is not None else 500.0))
    with np.load(path, allow_pickle=False) as z:
        required = {'contact_f_L', 'contact_f_R', 'contact_pos_L', 'contact_pos_R', 'substep'}
        missing = sorted(required - set(z.files))
        if missing:
            return dict(events=[], coverage_reasons=['missing_fields:' + ','.join(missing)], metadata=meta, sampling='unavailable')
        f = np.minimum(z['contact_f_L'], z['contact_f_R']).astype(float)
        (L, R) = (z['contact_pos_L'], z['contact_pos_R'])
        steps = z['substep']
        held = z['grasped'].astype(bool) if 'grasped' in z.files else None
        if not len(L) == len(R) == len(steps) == len(f):
            raise ValueError(f'inconsistent trajectory lengths: {path}')
        direct = 'contact_time_s' in z.files
        times = z['contact_time_s'] if direct else steps / hz
        d = R - L
        lengths = np.linalg.norm(d, axis=1)
        valid = np.isfinite(f) & np.isfinite(L).all(axis=1) & np.isfinite(R).all(axis=1) & (lengths > 0)
        on = valid & (f > 0)
        dirs = np.divide(d, lengths[:, None], out=np.zeros_like(d), where=lengths[:, None] > 0)
        n = max(1, int(np.ceil(sustain_s * hz)))
        cos_limit = np.cos(np.deg2rad(max_angle_deg))
        (groups, current) = ([], [])
        for i in range(len(f)):
            if not on[i]:
                if current:
                    groups.append(current)
                    current = []
                continue
            if current:
                a = current[0]
                stable = steps[i] == steps[current[-1]] + 1 and np.linalg.norm(L[i] - L[a]) <= max_drift_m and (np.linalg.norm(R[i] - R[a]) <= max_drift_m) and (float(dirs[i] @ dirs[a]) >= cos_limit)
                if not stable:
                    groups.append(current)
                    current = []
            current.append(i)
        if current:
            groups.append(current)
        (events, short) = ([], 0)
        for g in groups:
            (force, local) = max_sustained(f[g], n)
            if local is None:
                short += 1
                continue
            i = g[local]
            events.append(dict(i=int(i), start=int(g[0]), stop=int(g[-1]) + 1, time_s=float(times[i]), start_time_s=float(times[g[0]]), stop_time_s=float(times[g[-1]] + 1 / hz), pL=L[i].tolist(), pR=R[i].tolist(), fL=force, fR=force, grasped=None if held is None else bool(held[i]), context='squeeze_in_grasp' if held is not None and held[i] else 'pinch_without_grasp', contact_s=len(g) / hz, raw_peak=float(f[g].max()), sustained=True))
            events[-1]['geometry_frame'] = frame
        reasons = []
        if np.any(~np.isfinite(z['contact_f_L']) | ~np.isfinite(z['contact_f_R'])):
            reasons.append('missing_force_observations')
        if np.any((z['contact_f_L'] > 0) & (z['contact_f_R'] > 0) & ~valid):
            reasons.append('invalid_bilateral_contact_samples')
        if not events:
            reasons.append('no_eligible_stable_bilateral_grasp')
        return dict(events=events, coverage_reasons=reasons, metadata=meta, excluded=dict(short_or_moving_contact_groups=short), sampling='fresh_contact_timestamps' if direct else 'update_cadence_not_recorded', configuration=dict(sustain_s=sustain_s, sim_freq=hz, max_drift_m=max_drift_m, max_angle_deg=max_angle_deg))

def first_sustained_crossing(path, event, force_bound, sustain_s=SUSTAIN_S):
    with np.load(path, allow_pickle=False) as z:
        sl = slice(event['start'], event['stop'])
        f = np.minimum(z['contact_f_L'][sl], z['contact_f_R'][sl])
        steps = z['substep'][sl]
        hz = len(f) / event['contact_s']
        t = z['contact_time_s'][sl] if 'contact_time_s' in z.files else steps / hz
        n = int(np.ceil(sustain_s * hz))
        windows = np.lib.stride_tricks.sliding_window_view(f, n)
        indices = np.flatnonzero(windows.min(1) >= force_bound)
        if not len(indices):
            raise ValueError('declared damaged event has no qualifying force window')
        i = int(indices[0])
        return dict(start_s=float(t[i]), confirmed_s=float(t[i + n - 1]))

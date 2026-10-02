import numpy as np


def estimate(target, donors):
    points = np.asarray(target['points_m'], float)
    candidates = {}
    for donor in donors:
        if donor.get('origin') != 'direct' or donor['object_id'] != target['object_id']:
            continue
        if donor['compatibility'] != target['compatibility']:
            continue
        bounds = donor['threshold']
        if bounds['kind'] != 'bracketed':
            continue
        p = np.asarray(donor['points_m'], float)
        ident = tuple(p[np.lexsort((p[:, 2], p[:, 1], p[:, 0]))].round(9).ravel())
        distance = min(np.linalg.norm(points - p), np.linalg.norm(points - p[::-1])) / np.sqrt(2)
        if ident not in candidates:
            candidates[ident] = (distance, donor)
    selected = sorted(candidates.values(), key=lambda x: (x[0], x[1]['id']))[:5]
    if len(selected) != 5:
        return dict(kind='failed', lower=None, upper=None, reason='fewer_than_five_direct_donors',
                    available_donors=len(selected))
    values = [x['threshold']['estimate'] if x['threshold'].get('estimate') is not None
              else (x['threshold']['lower'] + x['threshold']['upper']) / 2 for _, x in selected]
    value = float(np.median(values))
    return dict(kind='neighbor_median', lower=value, upper=value, estimate=value,
                origin='estimated', donors=[dict(id=x['id'], distance_m=float(d)) for d, x in selected])

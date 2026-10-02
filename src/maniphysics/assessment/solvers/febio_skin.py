from functools import lru_cache
import numpy as np
from .febio_log import read_log
from ..core.reference_fields import surface_geometry, tensors


@lru_cache(maxsize=32)
def logs(directory):
    return ([dict(read_log(directory / f'skin_ip{i}.txt')) for i in range(21)],
            dict(read_log(directory / 'object.txt')))


def membrane_ratio(directory, time, case, geometry):
    skins, displacements = logs(directory)
    expected = np.asarray(case['skin']['expected_fields']['element_ids'])
    stress = []
    for stream in skins:
        block = stream[time]
        block = block[np.argsort(block[:, 0])]
        if not np.array_equal(block[:, 0], expected):
            raise ValueError('Incomplete skin integration-point output')
        stress.append(block[:, 1:7])
    stress = np.stack(stress, axis=1).reshape(-1, 3, 7, 6)
    displacement = displacements[time]
    displacement = displacement[np.argsort(displacement[:, 0])]
    if not np.array_equal(displacement[:, 0], case['object_nodes']):
        raise ValueError('Incomplete object displacement output')
    geo = surface_geometry(geometry['object_nodes'], displacement[:, 1:4], geometry['skin_elements'])
    nu = case['skin']['material']['nu']
    stretch2 = 1 - nu / (1 - nu) * (geo['trace_surface_C'] - 2)
    if np.any(stretch2 <= 0):
        raise ValueError('Skin thickness model left its admissible domain')
    thickness = case['skin']['thickness_m'] * np.sqrt(stretch2)
    mean = np.einsum('l,elpc->epc', np.array([5., 8., 5.]) / 18, stress)
    n = geo['normal']
    projector = np.eye(3) - n[:, :, None] * n[:, None, :]
    membrane = projector[:, None] @ tensors(mean) @ projector[:, None]
    resultant = np.linalg.eigvalsh(membrane)[..., -1] * thickness[:, None]
    return max(0., float(resultant.max())) / case['limit']

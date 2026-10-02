import numpy as np
from .board import board_ratio

class ReferenceDomainError(ValueError):
    pass

def tensors(stress):
    s = np.asarray(stress)
    t = np.zeros(s.shape[:-1] + (3, 3))
    (t[..., 0, 0], t[..., 1, 1], t[..., 2, 2]) = (s[..., 0], s[..., 1], s[..., 2])
    t[..., 0, 1] = t[..., 1, 0] = s[..., 3]
    t[..., 0, 2] = t[..., 2, 0] = s[..., 4]
    t[..., 1, 2] = t[..., 2, 1] = s[..., 5]
    return t

def shell_stresses(block):
    b = np.asarray(block)
    b = b[np.lexsort((b[:, 1], b[:, 0]))]
    if len(b) % 9 or not np.all(b[:, 1].reshape(-1, 9) == np.arange(1, 10)):
        raise ValueError('complete nine-IP S6 stress required')
    return b[:, 2:].reshape(-1, 3, 3, 6)

def surface_geometry(nodes, displacement, triangles, axis=None):
    tri = np.asarray(triangles)[:, :3]
    x = np.asarray(nodes)[tri]
    y = (np.asarray(nodes) + displacement)[tri]
    X = np.stack((x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), axis=-1)
    Y = np.stack((y[:, 1] - y[:, 0], y[:, 2] - y[:, 0]), axis=-1)
    G = np.einsum('eji,ejk->eik', X, X)
    inverse = np.linalg.inv(G)
    F = Y @ inverse @ X.swapaxes(-1, -2)
    n0 = np.cross(X[:, :, 0], X[:, :, 1])
    n0 /= np.linalg.norm(n0, axis=1)[:, None]
    normal = np.cross(Y[:, :, 0], Y[:, :, 1])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    trace_c = np.einsum('eij,eij->e', F, F)
    result = dict(normal=normal, trace_surface_C=trace_c)
    if axis is not None:
        a = np.asarray(axis, float)
        a /= np.linalg.norm(a)
        initial = a - np.einsum('ei,i->e', n0, a)[:, None] * n0
        if np.any(np.linalg.norm(initial, axis=1) < 1e-06):
            raise ValueError('declared material direction is normal to a shell element')
        first = np.einsum('eij,ej->ei', F, initial)
        first /= np.linalg.norm(first, axis=1)[:, None]
        second = np.cross(normal, first)
        result['basis'] = np.stack((first, second, normal), axis=-1)
    return result

def membrane_resultant(block, nodes, displacement, triangles, thickness, nu):
    stress = shell_stresses(block)
    geometry = surface_geometry(nodes, displacement, triangles)
    stretch2 = 1 - nu / (1 - nu) * (geometry['trace_surface_C'] - 2)
    if np.any(stretch2 <= 0):
        raise ReferenceDomainError('reference elastic skin thickness law left its admissible range')
    current_t = thickness * np.sqrt(stretch2)
    mean = np.einsum('l,elpc->epc', np.array([5.0, 8.0, 5.0]) / 18, stress)
    normal = geometry['normal']
    projection = np.eye(3) - normal[:, :, None] * normal[:, None, :]
    membrane = projection[:, None] @ tensors(mean) @ projection[:, None]
    values = np.linalg.eigvalsh(membrane)[..., -1] * current_t[:, None]
    return max(0.0, float(values.max()))

def paperboard_surface_ratio(block, nodes, displacement, triangles, axis):
    stress = shell_stresses(block)
    tri = np.asarray(triangles)[:, :3]
    x = np.asarray(nodes)[tri]
    n0 = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    n0 /= np.linalg.norm(n0, axis=1)[:, None]
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    keep = np.linalg.norm(a - np.einsum('ei,i->e', n0, a)[:, None] * n0, axis=1) >= 1e-06
    if not keep.any():
        raise ValueError('declared material direction is normal to every shell element')
    stress = stress[keep]
    triangles = np.asarray(triangles)[keep]
    center = stress.mean(axis=1)
    slope = (stress[:, 2] - stress[:, 0]) / (2 * np.sqrt(3 / 5))
    surface = tensors(np.stack((center - slope, center + slope), axis=1))
    basis = surface_geometry(nodes, displacement, triangles, axis)['basis'][:, None, None]
    local = basis.swapaxes(-1, -2) @ surface @ basis
    components = np.stack((local[..., 0, 0], local[..., 1, 1], local[..., 0, 1]), axis=-1)
    return float(board_ratio(components).max())

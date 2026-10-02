import numpy as np
from maniphysics.assessment.geometry import load_mesh

def quat2mat(q):
    (w, x, y, z) = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (x * w + y * z), 1 - 2 * (x * x + y * y)]])

def ray_hits(V, F, origin, direction, eps=1e-12):
    o = np.asarray(origin, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    (p0, p1, p2) = (V[F[:, 0]], V[F[:, 1]], V[F[:, 2]])
    (e1, e2) = (p1 - p0, p2 - p0)
    h = np.cross(d, e2)
    a = (e1 * h).sum(1)
    ok = np.abs(a) > eps
    f = np.zeros_like(a)
    f[ok] = 1.0 / a[ok]
    s = o - p0
    u = f * (s * h).sum(1)
    ok &= (u >= 0) & (u <= 1)
    q = np.cross(s, e1)
    v = f * (d * q).sum(1)
    ok &= (v >= 0) & (u + v <= 1)
    t = f * (e2 * q).sum(1)
    ok &= t > eps
    t = np.sort(t[ok])
    return o + t[:, None] * d

def bbox_offset(model_id):
    M = load_mesh(model_id)
    return np.asarray(M.bounds).mean(0)

def log_to_mesh_frame(model_id, p_log, *, frame=None):
    rotation = np.eye(3)
    if frame is not None:
        explicit = frame.get('mesh_in_object_quat_wxyz')
        platform = frame.get('platform') or ''
        robot = frame.get('robot_uid')
        if explicit is not None:
            q = np.asarray(explicit, float)
            if q.shape != (4,) or not np.isfinite(q).all() or (not np.isclose(q @ q, 1.0, atol=1e-06)):
                raise ValueError('mesh-in-object rotation must be a unit wxyz quaternion')
            rotation = quat2mat(q)
        elif platform.startswith('libero') or robot == 'panda':
            if model_id == 'chicken-egg':
                rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        elif robot != 'widowx':
            raise ValueError('unidentified logged body-to-asset frame')
    return rotation.T @ np.asarray(p_log, float) - bbox_offset(model_id)

def snap_to_surface(model_id, pts, close_axis):
    (V, F) = _mesh_local(model_id)
    d = np.asarray(close_axis, float)
    d = d / np.linalg.norm(d)
    back = np.asarray(pts, float).mean(0) - d * float(np.ptp(V, 0).max())
    h = ray_hits(V, F, back, d)
    return np.array([h[0], h[-1]]) if len(h) >= 2 else np.asarray(pts, float)

def _mesh_local(model_id):
    M = load_mesh(model_id)
    return (np.asarray(M.vertices) - np.asarray(M.bounds).mean(0), np.asarray(M.faces))

def two_point_symmetric(model_id, contact_pos, close_axis):
    M = load_mesh(model_id)
    V = np.asarray(M.vertices) - np.asarray(M.bounds).mean(0)
    F = np.asarray(M.faces)
    d = np.asarray(close_axis, float)
    d = d / np.linalg.norm(d)
    c = np.asarray(contact_pos, float)
    back = c - d * float(np.ptp(V, 0).max())
    hits = ray_hits(V, F, back, d)
    if len(hits) < 2:
        raise RuntimeError(f'Expected two surface intersections, found {len(hits)}')
    return np.array([hits[0], hits[-1]])

def close_axis_obj(tcp_q, obj_q, tcp_axis):
    return quat2mat(obj_q).T @ quat2mat(tcp_q)[:, tcp_axis]

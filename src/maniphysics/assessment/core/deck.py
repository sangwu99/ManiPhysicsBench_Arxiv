import json
from pathlib import Path
import numpy as np

def displacement_frame(nodes, avoid, r_avoid, direction):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    xyz = np.asarray(nodes, float)
    far = np.ones(len(xyz), bool)
    for p in np.atleast_2d(avoid):
        far &= np.linalg.norm(xyz - p, axis=1) > r_avoid
    ids = np.flatnonzero(far)
    if len(ids) < 2:
        raise ValueError('need two support nodes outside contact exclusion regions')
    q = xyz[ids] - xyz[ids].mean(axis=0)
    q -= (q @ d)[:, None] * d
    ia = int(np.argmax(np.linalg.norm(q, axis=1)))
    ib = int(np.argmax(np.linalg.norm(q - q[ia], axis=1)))
    e1 = q[ib] - q[ia]
    span = float(np.linalg.norm(e1))
    if span <= 1e-12:
        raise ValueError('transverse supports do not constrain rigid rotation')
    e1 /= span
    basis = np.stack([d, e1, np.cross(d, e1)])
    (a, b) = (int(ids[ia]) + 1, int(ids[ib]) + 1)
    return (basis, [(a, 2), (a, 3), (b, 3)], span)

def _mid_idx(k):
    return list(range({6: 3, 8: 4, 10: 4}[k], k))

def _areas(nodes, elems):
    c = elems[:, :4] - 1 if elems.shape[1] == 8 else elems[:, :3] - 1
    p = nodes[c]
    a = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    if c.shape[1] == 4:
        a += 0.5 * np.linalg.norm(np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=1)
    return a

def patch_loads(nodes, elems, pt, direction, F, c, inward_to=None):
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    if inward_to is not None:
        if float(np.dot(d, np.asarray(inward_to, float) - np.asarray(pt, float))) < 0:
            d = -d
    rel = nodes - np.asarray(pt, float)
    if np.isscalar(c):
        inside = np.linalg.norm(rel, axis=1) <= c
    else:
        (a, b) = c
        ez = np.array([0.0, 0.0, 1.0])
        ep = np.cross(ez, d)
        ep = ep / max(np.linalg.norm(ep), 1e-12)
        inside = (np.abs(rel @ ez) <= a) & (np.abs(rel @ ep) <= b) & (np.abs(rel @ d) <= max(a, b))
    w = inside[elems - 1].sum(1) / elems.shape[1]
    A = _areas(nodes, elems)
    eff = float((w * A).sum())
    if eff <= 0:
        raise RuntimeError(f'Non-positive effective contact area: {c}')
    p = F / eff
    mids = _mid_idx(elems.shape[1])
    acc = {}
    for (e, we, Ae) in zip(elems, w, A):
        if we <= 0:
            continue
        f = p * we * Ae / len(mids)
        for m in mids:
            n = int(e[m])
            acc[n] = acc.get(n, np.zeros(3)) + f * d
    return (acc, eff)

def patch_nodes(nodes, pt, direction, c):
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    rel = nodes - np.asarray(pt, float)
    if np.isscalar(c):
        m = np.linalg.norm(rel, axis=1) <= c
    else:
        (a, b) = c
        ez = np.array([0.0, 0.0, 1.0])
        ep = np.cross(ez, d)
        ep = ep / max(np.linalg.norm(ep), 1e-12)
        m = (np.abs(rel @ ez) <= a) & (np.abs(rel @ ep) <= b) & (np.abs(rel @ d) <= max(a, b))
    return np.where(m)[0] + 1

def _min_constraint_disp(nodes, avoid, r_avoid, d):
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    e1 = np.cross(d, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-06:
        e1 = np.cross(d, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    far = np.ones(len(nodes), bool)
    for a in np.atleast_2d(avoid):
        far &= np.linalg.norm(nodes - a, axis=1) > r_avoid
    idx = np.where(far)[0]
    if len(idx) < 2:
        idx = np.arange(len(nodes))
    Q = np.stack([nodes[idx] @ e1, nodes[idx] @ e2], 1)
    a = idx[np.argmax(np.linalg.norm(Q - Q.mean(0), axis=1))]
    b = idx[np.argmax(np.linalg.norm(Q - Q[list(idx).index(a)], axis=1))]
    ax = [int(np.argmax(np.abs(e1))) + 1, int(np.argmax(np.abs(e2))) + 1]
    return ([(a + 1, ax[0]), (a + 1, ax[1]), (b + 1, ax[1])], float(np.linalg.norm(nodes[b] - nodes[a])))

def _min_constraint(nodes, avoid, r_avoid):
    far = np.ones(len(nodes), bool)
    for a in np.atleast_2d(avoid):
        far &= np.linalg.norm(nodes - a, axis=1) > r_avoid
    idx = np.where(far)[0]
    if len(idx) < 3:
        idx = np.arange(len(nodes))
    Q = nodes[idx, :2]
    i0 = np.argmax(np.linalg.norm(Q - Q.mean(0), axis=1))
    i1 = np.argmax(np.linalg.norm(Q - Q[i0], axis=1))
    e = Q[i1] - Q[i0]
    i2 = np.argmax(np.abs(np.cross(Q - Q[i0], e)))
    tri = np.array([idx[i0], idx[i1], idx[i2]])
    area = 0.5 * abs(float(np.cross(Q[i1] - Q[i0], Q[i2] - Q[i0])))
    x = nodes[tri, 0]
    (a, b) = (tri[int(np.argmin(x))], tri[int(np.argmax(x))])
    c = int([t for t in tri if t not in (a, b)][0])
    return ([(a + 1, 1), (a + 1, 2), (a + 1, 3), (b + 1, 2), (b + 1, 3), (c + 1, 3)], area)

def write(path, nodes, elems, etype, *, loads, material, thickness=None, avoid=None, r_avoid=0.01, nlgeom=True, read_within=None, read_pts=None, skin=None, want_peeq=False, disp=None, ramp_inc=0.5):
    frame = np.eye(3)
    if disp:
        original_nodes = np.asarray(nodes, float)
        (frame, fixed, tri_area) = displacement_frame(original_nodes, avoid if avoid is not None else original_nodes[:1], r_avoid, disp[0][1])
        nodes = original_nodes @ frame.T
        avoid = None if avoid is None else np.asarray(avoid) @ frame.T
        read_pts = None if read_pts is None else np.asarray(read_pts) @ frame.T
        loads = {n: frame @ np.asarray(f) for (n, f) in loads.items()}
        disp = [(pn, frame @ np.asarray(dv), delta) for (pn, dv, delta) in disp]
        if any((np.linalg.norm(np.asarray(dv)[1:]) > 1e-08 for (_, dv, _) in disp)):
            raise ValueError('all prescribed contact directions must share one closing axis')
    L = ['*NODE, NSET=NALL']
    L += [f'{i + 1}, {p[0]:.9g}, {p[1]:.9g}, {p[2]:.9g}' for (i, p) in enumerate(nodes)]
    L.append(f'*ELEMENT, TYPE={etype}, ELSET=EALL')
    for (i, e) in enumerate(elems):
        L.append(f'{i + 1}, ' + ', '.join((str(int(v)) for v in e)))
    skin_ids = None
    if skin is not None:
        n0 = len(elems)
        L.append('*ELEMENT, TYPE=S6, ELSET=ESKIN')
        for (i, e) in enumerate(skin['elems']):
            L.append(f'{n0 + i + 1}, ' + ', '.join((str(int(v)) for v in e)))
        skin_ids = np.arange(n0 + 1, n0 + 1 + len(skin['elems']))

    def mat(name, m):
        out = [f'*MATERIAL, NAME={name}', '*ELASTIC', f"{m['E']:.9g}, {m['nu']:.9g}"]
        if m.get('plastic'):
            out.append('*PLASTIC')
            out += [f'{a:.9g}, {b:.9g}' for (a, b) in m['plastic']]
        return out
    L += mat('MAT', material)
    if etype.startswith('S'):
        L += ['*SHELL SECTION, ELSET=EALL, MATERIAL=MAT', f'{thickness:.9g}']
    else:
        L.append('*SOLID SECTION, ELSET=EALL, MATERIAL=MAT')
    if skin is not None:
        L += mat('MATSKIN', skin['material'])
        L += ['*SHELL SECTION, ELSET=ESKIN, MATERIAL=MATSKIN', f"{skin['thickness']:.9g}"]
    if not disp:
        (fixed, tri_area) = _min_constraint(nodes, avoid if avoid is not None else nodes[:1], r_avoid)
    L.append('*NSET, NSET=NFIX')
    L.append(', '.join((str(n) for n in sorted({n for (n, _) in fixed}))))
    L.append('*BOUNDARY')
    for (n, dof) in fixed:
        L.append(f'{n}, {dof}, {dof}')
    read_set = 'EALL'
    read_ids = np.arange(1, len(elems) + 1)
    rp = read_pts if read_pts is not None else avoid
    if read_within is not None and rp is not None:
        ctr = nodes[elems - 1].mean(1)
        near = np.zeros(len(elems), bool)
        for a in np.atleast_2d(rp):
            near |= np.linalg.norm(ctr - a, axis=1) <= read_within
        if near.sum() < 10:
            raise RuntimeError(f'Too few elements near the reference points: {int(near.sum())}')
        read_set = 'EREAD'
        read_ids = np.flatnonzero(near) + 1
        L.append('*ELSET, ELSET=EREAD')
        ids = [str(i + 1) for i in np.where(near)[0]]
        L += [', '.join(ids[k:k + 12]) for k in range(0, len(ids), 12)]
    skin_set = 'ESKIN'
    skin_read_ids = skin_ids
    if skin is not None and read_within is not None and (rp is not None):
        sc = nodes[skin['elems'] - 1].mean(1)
        sn = np.zeros(len(skin['elems']), bool)
        for a in np.atleast_2d(rp):
            sn |= np.linalg.norm(sc - a, axis=1) <= read_within
        if sn.sum() >= 5:
            skin_set = 'ESKINR'
            skin_read_ids = skin_ids[sn]
            L.append('*ELSET, ELSET=ESKINR')
            sid = [str(v) for v in skin_ids[sn]]
            L += [', '.join(sid[k:k + 12]) for k in range(0, len(sid), 12)]
    if disp:
        L.append('*NSET, NSET=NDRIVE')
        ids = [str(int(n)) for n in sorted(disp[0][0])]
        L += [', '.join(ids[k:k + 12]) for k in range(0, len(ids), 12)]
        if len(disp) == 2:
            L.append('*NSET, NSET=NDRIVE_OTHER')
            ids = [str(int(n)) for n in sorted(disp[1][0])]
            L += [', '.join(ids[k:k + 12]) for k in range(0, len(ids), 12)]
    L.append(f"*STEP, NLGEOM{(', INC=500' if nlgeom else '')}" if nlgeom else '*STEP')
    L += ['*STATIC', f'{ramp_inc:.4g}, 1.0, 1e-6, {ramp_inc:.4g}']
    if disp:
        L.append('*BOUNDARY')
        for (pn, dvec, delta) in disp:
            dv = np.asarray(dvec, float)
            dv = dv / np.linalg.norm(dv)
            for n in sorted(pn):
                L.append(f'{int(n)}, 1, 1, {delta * dv[0]:.9g}')
    else:
        L.append('*CLOAD')
        for (n, f) in sorted(loads.items()):
            for k in range(3):
                if abs(f[k]) > 0:
                    L.append(f'{n}, {k + 1}, {f[k]:.9g}')
    fq = ', FREQUENCY=1' if disp else ''
    L.append(f'*EL PRINT, ELSET={read_set}, GLOBAL=YES{fq}')
    L.append('S, PEEQ' if want_peeq else 'S')
    if skin is not None:
        L += [f'*EL PRINT, ELSET={skin_set}, GLOBAL=YES{fq}', 'S']
    L += [f'*NODE PRINT, NSET=NFIX{fq}', 'RF']
    if disp:
        L += [f'*NODE PRINT, NSET=NDRIVE{fq}', 'RF']
        if len(disp) == 2:
            L += [f'*NODE PRINT, NSET=NDRIVE_OTHER{fq}', 'RF']
    L += ['*NODE FILE', 'U', '*EL FILE', 'S', '*END STEP']
    with open(path, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    Path(path).with_suffix('.frame.json').write_text(json.dumps({'world_to_deck': frame.tolist(), 'reaction_frame': 'deck', 'isotropic_only': True, 'fixed_dofs': [[int(n), int(dof)] for (n, dof) in fixed]}))
    output_ids = {read_set: read_ids.tolist()}
    output_types = {read_set: etype}
    if skin is not None:
        output_ids[skin_set] = skin_read_ids.tolist()
        output_types[skin_set] = 'S6'
    reaction_nodes = {}
    if disp:
        reaction_nodes['NDRIVE'] = sorted((int(n) for n in disp[0][0]))
        if len(disp) == 2:
            reaction_nodes['NDRIVE_OTHER'] = sorted((int(n) for n in disp[1][0]))
    return dict(n_node=len(nodes), n_elem=len(elems), fixed=fixed, output_element_ids=output_ids, output_element_types=output_types, output_reaction_nodes=reaction_nodes, fix_tri_area=tri_area, n_read=len(elems) if read_set == 'EALL' else int(near.sum()))

import numpy as np
TET10_G2C = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
C3D10_FACES = {1: (0, 1, 2), 2: (0, 3, 1), 3: (1, 3, 2), 4: (2, 3, 0)}
PAD = dict(E=20000000.0, nu=0.48)
PAD_R = 0.008463
PAD_H = 0.002

def cyl_mesh(R=PAD_R, h=PAD_H, hz=0.0005, hr=0.0009):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber('General.NumThreads', 2)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, h, R)
        gmsh.model.occ.synchronize()
        f = gmsh.model.mesh.field.add('MathEval')
        gmsh.model.mesh.field.setString(f, 'F', f'{hr:.6g} + {max(hz, 1e-09):.6g}*z/{h:.6g}*3')
        gmsh.model.mesh.field.setAsBackgroundMesh(f)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
        gmsh.option.setNumber('Mesh.Optimize', 1)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber('Mesh.SecondOrderLinear', 1)
        gmsh.model.mesh.setOrder(2)
        (tg, tn, _) = gmsh.model.mesh.getNodes()
        idx = {int(t): i for (i, t) in enumerate(tg)}
        V = np.array(tn).reshape(-1, 3)
        (et, _, en) = gmsh.model.mesh.getElements(3)
        k = list(et).index(11)
        E = np.array(en[k], int).reshape(-1, 10)
        return (V, np.vectorize(idx.get)(E)[:, TET10_G2C])
    finally:
        gmsh.finalize()

def sphere_mesh(R, cap_frac=0.55, h=0.0012, h_fine=None, r_fine=0.004):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber('General.NumThreads', 2)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        s = gmsh.model.occ.addSphere(0, 0, 0, R)
        b = gmsh.model.occ.addBox(-2 * R, -2 * R, -2 * R, 4 * R, 4 * R, 2 * R * (1 - cap_frac))
        gmsh.model.occ.cut([(3, s)], [(3, b)])
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
        if h_fine:
            pf = gmsh.model.mesh.field.add('Distance')
            pt = gmsh.model.geo.addPoint(0, 0, R, h_fine)
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.field.setNumbers(pf, 'PointsList', [pt])
            tf = gmsh.model.mesh.field.add('Threshold')
            gmsh.model.mesh.field.setNumber(tf, 'InField', pf)
            gmsh.model.mesh.field.setNumber(tf, 'SizeMin', h_fine)
            gmsh.model.mesh.field.setNumber(tf, 'SizeMax', h)
            gmsh.model.mesh.field.setNumber(tf, 'DistMin', r_fine)
            gmsh.model.mesh.field.setNumber(tf, 'DistMax', r_fine * 3)
            gmsh.model.mesh.field.setAsBackgroundMesh(tf)
        else:
            gmsh.option.setNumber('Mesh.MeshSizeMin', h)
            gmsh.option.setNumber('Mesh.MeshSizeMax', h * 3)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber('Mesh.SecondOrderLinear', 1)
        gmsh.model.mesh.setOrder(2)
        (tg, tn, _) = gmsh.model.mesh.getNodes()
        idx = {int(t): i for (i, t) in enumerate(tg)}
        V = np.array(tn).reshape(-1, 3)
        (et, _, en) = gmsh.model.mesh.getElements(3)
        k = list(et).index(11)
        E = np.array(en[k], int).reshape(-1, 10)
        return (V, np.vectorize(idx.get)(E)[:, TET10_G2C])
    finally:
        gmsh.finalize()

def bnd_faces(E):
    rec = []
    for (ei, e) in enumerate(E):
        for (fid, cn) in C3D10_FACES.items():
            rec.append((ei, fid, tuple(sorted((int(e[c]) for c in cn)))))
    from collections import Counter
    cnt = Counter((r[2] for r in rec))
    return [r for r in rec if cnt[r[2]] == 1]

def _mat(name, m):
    if 'engineering_constants' in m:
        values = m['engineering_constants']
        if len(values) != 9 or m.get('plastic'):
            raise ValueError('orthotropic elastic predictor requires nine constants and no isotropic plastic law')
        out = [f'*MATERIAL, NAME={name}', '*ELASTIC, TYPE=ENGINEERING CONSTANTS', ','.join((f'{x:.9g}' for x in values[:8])), f'{values[8]:.9g}']
    else:
        out = [f'*MATERIAL, NAME={name}', '*ELASTIC', f"{m['E']:.9g}, {m['nu']:.9g}"]
    if m.get('plastic'):
        out.append('*PLASTIC')
        out += [f'{a:.9g}, {b:.9g}' for (a, b) in m['plastic']]
    return out

def _frame(d):
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, d)
    u /= np.linalg.norm(u)
    return (u, np.cross(d, u), d)

def place_pad(Vp, pt, d, gap=2e-05):
    (u, v, w) = _frame(d)
    R = np.stack([u, -v, -w], 1)
    assert np.linalg.det(R) > 0
    return Vp @ R.T + (np.asarray(pt, float) - w * gap)

def support_plane(nodes, elems, point, direction, radius=PAD_R):
    point = np.asarray(point, float)
    (u, v, d) = _frame(direction)
    q = np.asarray(nodes, float) - point
    local = np.stack([q @ u, q @ v, q @ d], axis=1)
    elems = np.asarray(elems)
    faces = [f[2] for f in bnd_faces(elems)] if elems.shape[1] == 10 else elems[:, :3]
    heights = []
    tol = 1e-10
    for ids in faces:
        tri = local[list(ids)]
        (xy, z) = (tri[:, :2], tri[:, 2])
        if np.any(xy.min(0) > radius) or np.any(xy.max(0) < -radius):
            continue
        heights.extend(z[np.sum(xy * xy, axis=1) <= radius ** 2 * (1 + tol)].tolist())
        for j in range(3):
            k = (j + 1) % 3
            a = xy[j]
            edge = xy[k] - a
            aa = float(edge @ edge)
            if aa == 0:
                continue
            bb = 2 * float(a @ edge)
            cc = float(a @ a - radius ** 2)
            disc = bb * bb - 4 * aa * cc
            if disc < 0:
                continue
            for t in ((-bb - np.sqrt(disc)) / (2 * aa), (-bb + np.sqrt(disc)) / (2 * aa)):
                if -tol <= t <= 1 + tol:
                    heights.append(float(z[j] + np.clip(t, 0, 1) * (z[k] - z[j])))
        mat = (xy[1:] - xy[0]).T
        if abs(np.linalg.det(mat)) <= 1e-14 * radius ** 2:
            continue
        grad = np.linalg.solve(mat.T, z[1:] - z[0])
        mag = np.linalg.norm(grad)
        candidate = -radius * grad / mag if mag > 0 else np.zeros(2)
        bary = np.linalg.solve(mat, candidate - xy[0])
        if np.all(bary >= -tol) and bary.sum() <= 1 + tol:
            heights.append(float(z[0] + grad @ (candidate - xy[0])))
    if not heights:
        raise ValueError('pad footprint does not intersect the remeshed surface')
    shift = min(heights)
    return (point + d * shift, dict(normal_translation_m=shift, tangential_translation_m=0.0))

def write(path, obj, pads, *, mat_obj, mat_pad, skin=None, delta, read_pts, read_within, fixed, ramp_inc=0.05, want_peeq=True, k_contact=None, inc0=None, mu=0.5, exclude_nodes=None, t_obj=None, contact_type='SURFACE TO SURFACE', rigid_pads=False):
    (Vo, Eo) = obj
    shell_obj = np.asarray(Eo).shape[1] in (3, 6)
    if contact_type not in ('SURFACE TO SURFACE', 'MORTAR'):
        raise ValueError(f'unsupported contact formulation: {contact_type}')
    if contact_type == 'MORTAR' and (shell_obj or skin is not None):
        raise ValueError('Mortar contact requires 3D solid elements')
    (L, nid, eid) = (['*NODE, NSET=NALL'], 0, 0)
    for p in Vo:
        nid += 1
        L.append(f'{nid}, {p[0]:.9g}, {p[1]:.9g}, {p[2]:.9g}')
    n_obj = nid
    (pad_no, pad_eo) = ([], [])
    for (Vp, Ep, d) in pads:
        pad_no.append(nid)
        for p in Vp:
            nid += 1
            L.append(f'{nid}, {p[0]:.9g}, {p[1]:.9g}, {p[2]:.9g}')
    L.append(f"*ELEMENT, TYPE={('S6' if shell_obj else 'C3D10')}, ELSET=EOBJ")
    for e in Eo:
        eid += 1
        L.append(f'{eid}, ' + ', '.join((str(int(v) + 1) for v in e)))
    n_eobj = eid
    for (k, (Vp, Ep, d)) in enumerate(pads):
        pad_eo.append(eid)
        L.append(f'*ELEMENT, TYPE=C3D10, ELSET=EPAD{k}')
        for e in Ep:
            eid += 1
            L.append(f'{eid}, ' + ', '.join((str(int(v) + 1 + pad_no[k]) for v in e)))
    skin_ids = None
    if skin is not None:
        L.append('*ELEMENT, TYPE=S6, ELSET=ESKIN')
        s0 = eid
        for e in skin['elems']:
            eid += 1
            L.append(f'{eid}, ' + ', '.join((str(int(v)) for v in e)))
        skin_ids = np.arange(s0 + 1, eid + 1)
    L += _mat('MOBJ', mat_obj)
    L += ['*SHELL SECTION, ELSET=EOBJ, MATERIAL=MOBJ', f'{t_obj:.9g}'] if shell_obj else ['*SOLID SECTION, ELSET=EOBJ, MATERIAL=MOBJ']
    L += _mat('MPAD', mat_pad)
    for k in range(len(pads)):
        L.append(f'*SOLID SECTION, ELSET=EPAD{k}, MATERIAL=MPAD')
    if skin is not None:
        L += _mat('MSKIN', skin['material'])
        L += ['*SHELL SECTION, ELSET=ESKIN, MATERIAL=MSKIN', f"{skin['thickness']:.9g}"]
    L.append('*SURFACE, NAME=SOBJ, TYPE=ELEMENT')
    if shell_obj:
        ctr_o = Vo[np.asarray(Eo)[:, :3]].mean(1)
        near = np.zeros(len(Eo), bool)
        for a in np.atleast_2d(read_pts):
            near |= np.linalg.norm(ctr_o - a, axis=1) <= max(2.0 * PAD_R, 0.012)
        for i in np.flatnonzero(near):
            L.append(f'{i + 1}, S1')
            L.append(f'{i + 1}, S2')
    else:
        ob = bnd_faces(Eo)
        ctr_o = np.array([Vo[list(f[2])].mean(0) for f in ob])
        near = np.zeros(len(ob), bool)
        for a in np.atleast_2d(read_pts):
            near |= np.linalg.norm(ctr_o - a, axis=1) <= max(2.0 * PAD_R, 0.012)
        for (f, ok) in zip(ob, near):
            if ok:
                L.append(f'{f[0] + 1}, S{f[1]}')
    for (k, (Vp, Ep, d)) in enumerate(pads):
        pb = bnd_faces(Ep)
        L.append(f'*SURFACE, NAME=SPAD{k}, TYPE=ELEMENT')
        proj = Vp @ np.asarray(d, float)
        on_bot = proj > proj.max() - 1e-09
        n_sel = 0
        for f in pb:
            if rigid_pads or all((on_bot[i] for i in f[2])):
                L.append(f'{pad_eo[k] + f[0] + 1}, S{f[1]}')
                n_sel += 1
        if n_sel < 50:
            raise RuntimeError(f'Too few pad contact faces for pad {k}: {n_sel}')
    kc = k_contact or 100.0 * mat_obj['E'] / 0.001
    L += ['*SURFACE INTERACTION, NAME=SI', '*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR', f'{kc:.6g}']
    if mu:
        L += ['*FRICTION', f'{mu:.4g}, {0.05 * kc:.6g}']
    for k in range(len(pads)):
        L += [f'*CONTACT PAIR, INTERACTION=SI, TYPE={contact_type}', f'SPAD{k}, SOBJ']
    L.append('*BOUNDARY')
    for (n, dof) in fixed:
        L.append(f'{n}, {dof}, {dof}')
    back = []
    for (k, (Vp, Ep, d)) in enumerate(pads):
        proj = Vp @ np.asarray(d, float)
        sel = np.arange(len(Vp)) if rigid_pads else np.flatnonzero(proj <= proj.min() + 1e-09)
        ids = [int(i) + 1 + pad_no[k] for i in sel]
        back.append(ids)
        L.append(f'*NSET, NSET=NBACK{k}')
        L += [', '.join((str(v) for v in ids[j:j + 12])) for j in range(0, len(ids), 12)]
    ctr = Vo[np.asarray(Eo)[:, :3] if shell_obj else Eo].mean(1)
    rd = np.zeros(len(Eo), bool)
    for a in np.atleast_2d(read_pts):
        rd |= np.linalg.norm(ctr - a, axis=1) <= read_within
    if exclude_nodes:
        ex = np.isin(Eo + 1, np.asarray(exclude_nodes)).any(1)
        rd &= ~ex
    if rd.sum() < 10:
        raise RuntimeError(f'Too few object elements near the contact: {int(rd.sum())}')
    L.append('*ELSET, ELSET=EREAD')
    ids = [str(i + 1) for i in np.flatnonzero(rd)]
    L += [', '.join(ids[j:j + 12]) for j in range(0, len(ids), 12)]
    if skin is not None:
        sc = Vo[np.asarray(skin['elems']) - 1].mean(1)
        sn = np.zeros(len(skin['elems']), bool)
        for a in np.atleast_2d(read_pts):
            sn |= np.linalg.norm(sc - a, axis=1) <= read_within
        if sn.sum() >= 5:
            L.append('*ELSET, ELSET=ESKINR')
            sid = [str(v) for v in skin_ids[sn]]
            L += [', '.join(sid[j:j + 12]) for j in range(0, len(sid), 12)]
    i0 = inc0 or ramp_inc / 20.0
    L += ['*STEP, NLGEOM, INC=2000', '*STATIC', f'{i0:.4g}, 1.0, 1e-9, {ramp_inc:.4g}', '*BOUNDARY']
    for (k, (Vp, Ep, d)) in enumerate(pads):
        dv = np.asarray(d, float)
        for j in range(3):
            L.append(f'NBACK{k}, {j + 1}, {j + 1}, {delta * dv[j]:.9g}')
    fq = ', FREQUENCY=1'
    L += [f'*EL PRINT, ELSET=EREAD, GLOBAL=YES{fq}', 'S, PEEQ' if want_peeq else 'S']
    if skin is not None:
        L += [f'*EL PRINT, ELSET=ESKINR, GLOBAL=YES{fq}', 'S']
    for k in range(len(pads)):
        L += [f'*NODE PRINT, NSET=NBACK{k}{fq}', 'RF']
    L += [f'*CONTACT PRINT{fq}', 'CSTR', '*NODE FILE', 'U', '*END STEP']
    open(path, 'w').write('\n'.join(L) + '\n')
    return dict(n_node=nid, n_elem=eid, n_obj=n_obj, n_eobj=n_eobj, pad_no=pad_no, pad_eo=pad_eo, back=back, contact_type=contact_type)

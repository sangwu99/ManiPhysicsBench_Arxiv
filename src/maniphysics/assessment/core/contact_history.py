from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from . import deck, pad_contact as pc, run, shell_surface
from .assessment import threshold_bounds
from .quality import reaction_quality
VERSION = 'deformable-pad-history-v2-reference-criteria'
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

@dataclass(frozen=True)
class Pad:
    radius_m: float = 0.008463
    thickness_m: float = 0.002
    crown_radius_m: float = 0.03
    E_Pa: float = 20000000.0
    nu: float = 0.48

    def validate(self):
        (r, h, c) = (self.radius_m, self.thickness_m, self.crown_radius_m)
        if not (np.isfinite(list(asdict(self).values())).all() and 0 < r < c and (h > c - np.sqrt(c * c - r * r)) and (self.E_Pa > 0) and (0 <= self.nu < 0.5)):
            raise ValueError('pad requires finite positive thickness at its rim and admissible elastic constants')
PLASTIC_VOLUME_FLOOR = 0.002
REACTION_ABSOLUTE_N = 0.001

def _plastic_volume_fraction(peeq_rows, geometry):
    nodes = np.asarray(geometry['nodes'], float)
    el = np.asarray(geometry['object_elements'], int)
    P = nodes[el[:, :4]]
    vol = np.abs(np.einsum('ij,ij->i', P[:, 1] - P[:, 0], np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]))) / 6
    e = peeq_rows[:, 0].astype(int) - 1
    ok = (e >= 0) & (e < len(el))
    total = np.zeros(len(el))
    count = np.zeros(len(el))
    np.add.at(total, e[ok], peeq_rows[ok, 2])
    np.add.at(count, e[ok], 1)
    return float(vol[total / np.maximum(count, 1) >= PLASTIC_VOLUME_FLOOR].sum() / vol.sum())

def _real(value):
    text = f'{float(value):.12g}'
    return text if any((c in text for c in '.eE')) else text + '.'

def _gmsh_tets(gmsh):
    (tags, coords, _) = gmsh.model.mesh.getNodes()
    index = {int(n): i for (i, n) in enumerate(tags)}
    (types, _, nodes) = gmsh.model.mesh.getElements(3)
    k = list(types).index(11)
    cells = np.array([[index[int(n)] for n in e] for e in np.asarray(nodes[k]).reshape(-1, 10)])
    return (np.asarray(coords).reshape(-1, 3), cells[:, pc.TET10_G2C])

def _mesh_sizes(gmsh, size_m, contact_size_m, points, refine_radius_m):
    gmsh.option.setNumber('Mesh.MeshSizeMin', contact_size_m or size_m)
    gmsh.option.setNumber('Mesh.MeshSizeMax', size_m)
    if contact_size_m is not None:
        if not 0 < contact_size_m <= size_m or refine_radius_m <= 0:
            raise ValueError('invalid declared contact refinement sizes')
        fields = []
        for point in points:
            field = gmsh.model.mesh.field.add('MathEval')
            distance = 'Sqrt(' + '+'.join((f'({axis}-({v:.12g}))^2' for (axis, v) in zip('xyz', point))) + ')'
            gmsh.model.mesh.field.setString(field, 'F', f'Min({size_m:.12g},{contact_size_m:.12g}+0.6*Max(0,{distance}-{refine_radius_m:.12g}))')
            fields.append(field)
        field = gmsh.model.mesh.field.add('Min')
        gmsh.model.mesh.field.setNumbers(field, 'FieldsList', fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(field)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)

def pad_mesh(spec=Pad(), *, size_m=0.001, contact_size_m=None, refine_radius_m=0.0006):
    import gmsh
    spec.validate()
    if not np.isfinite(size_m) or size_m <= 0:
        raise ValueError('positive pad mesh size required')
    gmsh.initialize()
    gmsh.option.setNumber('General.NumThreads', 2)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, spec.thickness_m, spec.radius_m)
        sphere = gmsh.model.occ.addSphere(0, 0, spec.crown_radius_m, spec.crown_radius_m)
        gmsh.model.occ.intersect([(3, cylinder)], [(3, sphere)])
        gmsh.model.occ.synchronize()
        _mesh_sizes(gmsh, size_m, contact_size_m, [(0, 0, 0)], refine_radius_m)
        gmsh.option.setNumber('Mesh.SecondOrderLinear', 0)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)
        return _gmsh_tets(gmsh)
    finally:
        gmsh.finalize()

def sphere_mesh(radius_m, *, size_m, contact_size_m=None, refine_radius_m=0.0006, partition_symmetry=False):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber('General.NumThreads', 2)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        if partition_symmetry:
            volumes = [(3, gmsh.model.occ.addSphere(0, 0, 0, radius_m, angle1=0, angle2=np.pi / 2, angle3=np.pi / 2))]
            for axis in np.eye(3):
                copy = gmsh.model.occ.copy(volumes)
                gmsh.model.occ.mirror(copy, *axis, 0.0)
                volumes += copy
            gmsh.model.occ.removeAllDuplicates()
        else:
            gmsh.model.occ.addSphere(0, 0, 0, radius_m)
        gmsh.model.occ.synchronize()
        _mesh_sizes(gmsh, size_m, contact_size_m, [(0, 0, -radius_m), (0, 0, radius_m)], refine_radius_m)
        gmsh.option.setNumber('Mesh.SecondOrderLinear', 0)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)
        return _gmsh_tets(gmsh)
    finally:
        gmsh.finalize()

def crown_support_plane(nodes, cells, point, direction, pad=Pad(), *, shell_thickness_m=None):
    from scipy.optimize import brentq
    if shell_thickness_m is not None:
        xyz = np.asarray(nodes, float)
        elements = np.asarray(cells, int)
        if elements.shape[1] != 6 or shell_thickness_m <= 0:
            raise ValueError('positive S6 shell thickness required for physical contact clearance')
        normal = np.cross(xyz[elements[:, 1]] - xyz[elements[:, 0]], xyz[elements[:, 2]] - xyz[elements[:, 0]])
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        vertex_normal = np.zeros_like(xyz)
        for j in range(6):
            np.add.at(vertex_normal, elements[:, j], normal)
        used = np.unique(elements)
        lengths = np.linalg.norm(vertex_normal[used], axis=1)
        if np.any(lengths <= 1e-12):
            raise ValueError('shell winding does not define a nodal surface normal')
        vertex_normal[used] /= lengths[:, None]
        outer = np.vstack((xyz + shell_thickness_m / 2 * vertex_normal, xyz - shell_thickness_m / 2 * vertex_normal))
        surfaces = np.vstack((elements, elements + len(xyz)))
        (anchor, info) = crown_support_plane(outer, surfaces, point, direction, pad)
        info.update(shell_thickness_m=shell_thickness_m)
        return (anchor, info)
    point = np.asarray(point, float)
    (u, v, d) = pc._frame(direction)
    q = np.asarray(nodes) - point
    local = np.column_stack((q @ u, q @ v, q @ d))
    cells = np.asarray(cells)
    faces = [f[2] for f in pc.bnd_faces(cells)] if cells.shape[1] == 10 else cells[:, :3]
    radius = pad.radius_m
    crown = pad.crown_radius_m
    values = []

    def sag(x):
        return crown - np.sqrt(crown * crown - float(x @ x))
    for ids in faces:
        tri = local[list(ids)]
        (xy, z) = (tri[:, :2], tri[:, 2])
        if np.any(xy.min(0) > radius) or np.any(xy.max(0) < -radius):
            continue
        for j in range(3):
            a = xy[j]
            edge = xy[(j + 1) % 3] - a
            dz = z[(j + 1) % 3] - z[j]
            aa = float(edge @ edge)
            if aa == 0:
                continue
            bb = 2 * float(a @ edge)
            cc = float(a @ a - radius * radius)
            disc = bb * bb - 4 * aa * cc
            if disc < 0:
                continue
            lo = max(0.0, (-bb - np.sqrt(disc)) / (2 * aa))
            hi = min(1.0, (-bb + np.sqrt(disc)) / (2 * aa))
            if lo > hi:
                continue
            candidates = [lo, hi]

            def derivative(t):
                x = a + t * edge
                return dz + float(x @ edge) / np.sqrt(crown * crown - float(x @ x))
            if derivative(lo) < 0 < derivative(hi):
                candidates.append(brentq(derivative, lo, hi))
            values.extend((float(z[j] + t * dz + sag(a + t * edge)) for t in candidates))
        mat = (xy[1:] - xy[0]).T
        if abs(np.linalg.det(mat)) <= 1e-16:
            continue
        gradient = np.linalg.solve(mat.T, z[1:] - z[0])
        candidate = -crown * gradient / np.sqrt(1 + float(gradient @ gradient))
        if np.linalg.norm(candidate) > radius:
            candidate *= radius / np.linalg.norm(candidate)
        bary = np.linalg.solve(mat, candidate - xy[0])
        if np.all(bary >= -1e-10) and bary.sum() <= 1 + 1e-10:
            values.append(float(z[0] + gradient @ (candidate - xy[0]) + sag(candidate)))
    if not values:
        raise ValueError('crown footprint does not intersect the object surface')
    shift = min(values)
    return (point + d * shift, dict(normal_translation_m=shift, tangential_translation_m=0.0))

def motion_arrays(motion):
    t = np.asarray(motion['times_s'], float)
    u = np.asarray(motion['translations_m'], float)
    rotations = np.asarray(motion.get('rotations', np.tile(np.eye(3), (len(t), 2, 1, 1))), float)
    if t.ndim != 1 or len(t) < 2 or u.shape != (len(t), 2, 3) or (rotations.shape != (len(t), 2, 3, 3)) or (t[0] != 0) or np.any(np.diff(t) <= 0) or (not np.isfinite(t).all()) or (not np.isfinite(u).all()) or (not np.isfinite(rotations).all()):
        raise ValueError('finite, increasing time and two jaw poses per time are required')
    if not np.allclose(rotations @ rotations.swapaxes(-1, -2), np.eye(3), atol=1e-10, rtol=0) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-10, rtol=0):
        raise ValueError('jaw rotations must be proper orthogonal matrices')
    if np.any(u[0] != 0) or not np.allclose(rotations[0], np.eye(3), atol=1e-12, rtol=0):
        raise ValueError('initial motion must match the supplied unloaded mesh pose')
    return (t, u, rotations)

def _set(lines, kind, name, ids):
    lines.append(f'*{kind}, {kind}={name}')
    ids = list(ids)
    lines.extend((','.join(map(str, ids[i:i + 12])) for i in range(0, len(ids), 12)))

def write_case(directory, body, pads, *, material, pad=Pad(), motion, criterion, limit, thickness_m=None, contact_penalty_Pa_m=200000000000.0, friction=0.5, centering_N_m=0.01, max_increment_s=0.01, source=None, diagnostic_symmetry_planes=(), contact_slave='pad', contact_type='SURFACE TO SURFACE', pad_contact_surface='exposed', skin=None, secondary_criterion=None, secondary_limit=None):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    pad.validate()
    (t, translations, rotations) = motion_arrays(motion)
    (xyz, cells) = (np.asarray(body[0], float), np.asarray(body[1], int))
    shell = cells.shape[1] == 6
    if cells.shape[1] not in (6, 10) or len(pads) != 2 or (not np.isfinite(xyz).all()):
        raise ValueError('one S6/C3D10 body and exactly two pads required')
    if shell and (thickness_m is None or thickness_m <= 0):
        raise ValueError('shell thickness required')
    if criterion not in ('peeq', 's1', 's3', 'shell_yield', 'skin_s1', 'skin_membrane', 'paperboard_yield', 'plastic_volume') or limit <= 0 or (not np.isfinite(limit)):
        raise ValueError('explicit supported initiation criterion and positive limit required')
    if criterion in ('shell_yield', 'paperboard_yield') and (not shell):
        raise ValueError('shell yield requires an S6 body')
    if criterion.startswith('skin_') != (skin is not None):
        raise ValueError('skin indicator and explicit attached skin must be supplied together')
    if skin is not None and (shell or material.get('engineering_constants') or (not contact_type.startswith('SURFACE TO SURFACE'))):
        raise ValueError('attached skin requires a solid body and surface-to-surface contact')
    if secondary_criterion is not None and (skin is None or secondary_criterion not in ('s1', 's3', 'peeq', 'plastic_volume') or (not secondary_limit) or (secondary_limit <= 0)):
        raise ValueError('secondary core criterion requires skin/core model and positive limit')
    if not (contact_penalty_Pa_m > 0 and centering_N_m > 0 and (max_increment_s > 0) and (friction >= 0)):
        raise ValueError('positive stiffnesses/increment and nonnegative friction required')
    if contact_slave not in ('pad', 'object'):
        raise ValueError('contact slave must be pad or object')
    if contact_type not in ('SURFACE TO SURFACE', 'MORTAR', 'SURFACE TO SURFACE, SMALL SLIDING', 'NODE TO SURFACE'):
        raise ValueError('unsupported explicit contact discretization')
    if contact_type == 'MORTAR' and contact_slave == 'object':
        raise ValueError('MORTAR requires distinct slave surfaces; the full object cannot be slave in both pairs')
    if pad_contact_surface not in ('exposed', 'crown'):
        raise ValueError('pad contact surface must be exposed or crown')
    directions = np.asarray([p[2] for p in pads], float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    force_history = motion.get('normal_forces_N')
    if force_history is not None:
        force_history = np.asarray(force_history, float)
        if force_history.shape != (len(t), 2) or not np.isfinite(force_history).all() or np.any(force_history < 0) or np.any(force_history[0] != 0):
            raise ValueError('normal force history must be nonnegative, finite and start unloaded')
        if not np.allclose(np.einsum('tkij,kj->tki', rotations, directions), directions, atol=1e-10, rtol=0):
            raise ValueError('force replay currently requires fixed closing directions')
        if np.max(abs(np.einsum('tki,ki->tk', translations, directions))) > 1e-10:
            raise ValueError('force replay prescribes only tangential translations; normal closure is solved')
        if shell and criterion in ('shell_yield', 'paperboard_yield'):
            raise ValueError('C-axis buckling requires pose/servo control; scalar force-controlled history is unsupported')
    all_xyz = [xyz]
    node_offsets = []
    element_offsets = []
    backs = []
    (nnode, nelem) = (len(xyz), len(cells))
    for (vertices, elements, direction) in pads:
        (vertices, elements) = (np.asarray(vertices), np.asarray(elements))
        node_offsets.append(nnode)
        element_offsets.append(nelem)
        projection = vertices @ (np.asarray(direction) / np.linalg.norm(direction))
        back = np.flatnonzero(projection <= projection.min() + 1e-09)
        if len(back) < 3:
            raise ValueError('pad backing must contain a planar bonded face')
        backs.append((back + 1 + nnode).tolist())
        nnode += len(vertices)
        nelem += len(elements)
        all_xyz.append(vertices)
    vertices = np.vstack(all_xyz)
    force_anchors = []
    force_refs = []
    if force_history is not None:
        for back in backs:
            positions = vertices[np.asarray(back) - 1]
            force_anchors.append(list(range(len(vertices) + 1, len(vertices) + len(back) + 1)))
            vertices = np.vstack((vertices, positions))
            force_refs.append(len(vertices) + 1)
            vertices = np.vstack((vertices, positions.mean(0)))
    lines = ['*HEADING', VERSION, '*NODE, NSET=NALL']
    lines.extend((f'{i + 1},' + ','.join((f'{v:.12g}' for v in p)) for (i, p) in enumerate(vertices)))
    lines.append(f"*ELEMENT, TYPE={('S6' if shell else 'C3D10')}, ELSET=EOBJ")
    lines.extend((f'{i + 1},' + ','.join(map(str, e + 1)) for (i, e) in enumerate(cells)))
    for (k, (_, elements, _)) in enumerate(pads):
        lines.append(f'*ELEMENT, TYPE=C3D10, ELSET=EPAD{k}')
        lines.extend((f'{element_offsets[k] + i + 1},' + ','.join(map(str, np.asarray(e) + node_offsets[k] + 1)) for (i, e) in enumerate(elements)))
    skin_ids = []
    if skin is not None:
        sk = np.asarray(skin['elements'], int)
        if sk.ndim != 2 or sk.shape[1] != 6 or sk.min() < 0 or (sk.max() >= len(xyz)) or (skin['thickness_m'] <= 0):
            raise ValueError('skin requires zero-based S6 connectivity on the body and positive thickness')
        skin_ids = list(range(nelem + 1, nelem + len(sk) + 1))
        nelem += len(sk)
        lines.append('*ELEMENT, TYPE=S6, ELSET=ESKIN')
        lines.extend((f'{eid},' + ','.join(map(str, e + 1)) for (eid, e) in zip(skin_ids, sk)))
    lines += pc._mat('OBJECT', material)
    orientation = ''
    if material.get('engineering_constants'):
        if not shell:
            raise ValueError('orthotropic reference support is restricted to shells')
        axis = np.asarray(material['orientation_axis'], float)
        axis /= np.linalg.norm(axis)
        other = np.eye(3)[np.argmin(abs(axis))]
        lines += ['*ORIENTATION, NAME=REFERENCE_AXES', ','.join((f'{v:.12g}' for v in [*axis, *other]))]
        orientation = ', ORIENTATION=REFERENCE_AXES'
    lines.append(f"*{('SHELL' if shell else 'SOLID')} SECTION, ELSET=EOBJ, MATERIAL=OBJECT" + orientation)
    if shell:
        lines.append(f'{thickness_m:.12g}')
    if skin is not None:
        lines += pc._mat('SKIN', skin['material'])
        lines += ['*SHELL SECTION, ELSET=ESKIN, MATERIAL=SKIN', f"{skin['thickness_m']:.12g}"]
    shear = pad.E_Pa / (2 * (1 + pad.nu))
    bulk = pad.E_Pa / (3 * (1 - 2 * pad.nu))
    lines += ['*MATERIAL, NAME=PAD', '*HYPERELASTIC, NEO HOOKE', f'{shear / 2:.12g},{2 / bulk:.12g}']
    for k in range(2):
        lines.append(f'*SOLID SECTION, ELSET=EPAD{k}, MATERIAL=PAD')
    lines.append('*SURFACE, NAME=SOBJ, TYPE=ELEMENT')
    if skin is not None:
        lines.extend((f'{eid},S{side}' for eid in skin_ids for side in (1, 2)))
    elif shell:
        lines.extend((f'{i + 1},S{side}' for i in range(len(cells)) for side in (1, 2)))
    else:
        lines.extend((f'{i + 1},S{side}' for (i, side, _) in pc.bnd_faces(cells)))
    for (k, (pad_vertices, elements, pad_direction)) in enumerate(pads):
        lines.append(f'*SURFACE, NAME=SPAD{k}, TYPE=ELEMENT')
        back = set(np.asarray(backs[k]) - node_offsets[k] - 1)
        faces = []
        for (i, side, face) in pc.bnd_faces(elements):
            if set(face).issubset(back):
                continue
            if pad_contact_surface == 'crown':
                corners = np.asarray(pad_vertices)[list(face)]
                normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
                minimum_cos = np.sqrt(1 - (pad.radius_m / pad.crown_radius_m) ** 2)
                if abs(normal @ directions[k]) < 0.5 * minimum_cos * np.linalg.norm(normal):
                    continue
            faces.append((i, side))
        if not faces:
            raise ValueError('empty deformable pad contact surface')
        lines.extend((f'{element_offsets[k] + i + 1},S{side}' for (i, side) in faces))
        _set(lines, 'NSET', f'NBACK{k}', backs[k])
    lines += ['*SURFACE INTERACTION, NAME=PADCONTACT', '*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR', f'{contact_penalty_Pa_m:.12g}']
    if friction:
        lines += ['*FRICTION', f'{friction:.12g},{0.05 * contact_penalty_Pa_m:.12g}']
    pairs = [(f'SPAD{k}', 'SOBJ') if contact_slave == 'pad' else ('SOBJ', f'SPAD{k}') for k in range(2)]
    for (slave, master) in pairs:
        lines += [f'*CONTACT PAIR, INTERACTION=PADCONTACT, TYPE={contact_type}', f'{slave},{master}']
    anchor = np.asarray([np.mean(p[0], axis=0) for p in pads])
    (fixed, _) = deck._min_constraint(xyz, anchor, 0.2 * np.linalg.norm(np.ptp(xyz, axis=0)))
    center_nodes = sorted({int(n) for (n, _) in fixed})
    _set(lines, 'NSET', 'NCENTER', center_nodes)
    _set(lines, 'NSET', 'NOBJECT', range(1, len(xyz) + 1))
    fixture_nodes = set()
    if diagnostic_symmetry_planes:
        if not source or source.get('kind') != 'numerical_diagnostic':
            raise ValueError('symmetry fixtures are reserved for explicitly declared numerical diagnostics')
        lines.append('*BOUNDARY')
        for axis in diagnostic_symmetry_planes:
            plane = np.flatnonzero(abs(xyz[:, axis]) < 1e-10) + 1
            if len(plane) < 3:
                raise ValueError('mesh must contain the declared symmetry plane')
            lines.extend((f'{node},{axis + 1},{axis + 1},0.' for node in plane))
            fixture_nodes.update(map(int, plane))
        _set(lines, 'NSET', 'NFIXTURE', sorted(fixture_nodes))
    for dof in (1, 2, 3):
        lines.append(f'*ELEMENT, TYPE=SPRING1, ELSET=ECENTER{dof}')
        for node in center_nodes:
            nelem += 1
            lines.append(f'{nelem},{node}')
        lines += [f'*SPRING, ELSET=ECENTER{dof}', str(dof), _real(centering_N_m)]
    if force_history is not None:
        for (k, ref) in enumerate(force_refs):
            _set(lines, 'NSET', f'NANCHOR{k}', force_anchors[k])
            _set(lines, 'NSET', f'NFORCE{k}', [ref])
            lines += ['*BOUNDARY', f'{ref},2,3,0.']
            nelem += 1
            lines += [f'*ELEMENT, TYPE=SPRING1, ELSET=EFORCE{k}', f'{nelem},{ref}', f'*SPRING, ELSET=EFORCE{k}', '1', _real(centering_N_m)]
            for (back, anchor) in zip(backs[k], force_anchors[k]):
                for (dof, coefficient) in enumerate(directions[k], 1):
                    terms = [f'{back},{dof},1.', f'{anchor},{dof},-1.']
                    if abs(coefficient) > 1e-14:
                        terms.append(f'{ref},1,{-coefficient:.12g}')
                    lines += ['*EQUATION', str(len(terms)), ','.join(terms)]
    boundaries = []
    amplitude_index = 0
    for k in range(2):
        back_xyz = vertices[np.asarray(backs[k]) - 1]
        center = back_xyz.mean(0)
        displacement = np.einsum('tij,nj->tni', rotations[:, k], back_xyz - center) + center + translations[:, k, None, :] - back_xyz
        for (ni, node) in enumerate(force_anchors[k] if force_history is not None else backs[k]):
            for dof in range(3):
                amplitude_index += 1
                name = f'A{amplitude_index}'
                lines.append(f'*AMPLITUDE, NAME={name}, TIME=TOTAL TIME')
                lines.extend((f'{ti:.12g},{ui:.12g}' for (ti, ui) in zip(t, displacement[:, ni, dof])))
                boundaries += [f'*BOUNDARY, AMPLITUDE={name}', f'{node},{dof + 1},{dof + 1},1.']
    force_loads = []
    if force_history is not None:
        for (k, ref) in enumerate(force_refs):
            lines.append(f'*AMPLITUDE, NAME=FN{k}, TIME=TOTAL TIME')
            lines.extend((f'{ti:.12g},{f:.12g}' for (ti, f) in zip(t, force_history[:, k])))
            force_loads += [f'*CLOAD, AMPLITUDE=FN{k}', f'{ref},1,1.']
    header = '\n'.join(lines) + '\n'
    steps = []
    constant_stages = []
    for (j, duration) in enumerate(np.diff(t)):
        unchanged = np.array_equal(translations[j], translations[j + 1]) and np.array_equal(rotations[j], rotations[j + 1]) and (force_history is None or np.array_equal(force_history[j], force_history[j + 1]))
        initial_increment = duration if unchanged else min(max_increment_s / 10, duration / 20)
        maximum_increment = duration if unchanged else min(max_increment_s, duration)
        if unchanged:
            constant_stages.append(j + 1)
        step = ['*STEP, NLGEOM, INC=10000', '*STATIC', f'{initial_increment:.12g},{duration:.12g},1e-10,{maximum_increment:.12g}']
        if j == 0:
            step += boundaries + force_loads
        step += ['*EL PRINT, ELSET=EOBJ, GLOBAL=YES, FREQUENCY=1', 'S,PEEQ' if material.get('plastic') else 'S', '*NODE PRINT, NSET=NCENTER, FREQUENCY=1', 'U', '*NODE PRINT, NSET=NOBJECT, FREQUENCY=1', 'U']
        if skin is not None:
            step += ['*EL PRINT, ELSET=ESKIN, GLOBAL=YES, FREQUENCY=1', 'S']
        for k in range(2):
            step += [f'*NODE PRINT, NSET=NBACK{k}, FREQUENCY=1', 'RF,U', f'*CONTACT PRINT, SLAVE={pairs[k][0]}, MASTER={pairs[k][1]}, FREQUENCY=1', 'CF,CFN,CFS,CSTR,CDIS' if contact_type.startswith('SURFACE TO SURFACE') else 'CSTR,CDIS']
            if force_history is not None:
                step += [f'*NODE PRINT, NSET=NANCHOR{k}, FREQUENCY=1', 'RF', f'*NODE PRINT, NSET=NFORCE{k}, FREQUENCY=1', 'U']
        if fixture_nodes:
            step += ['*NODE PRINT, NSET=NFIXTURE, FREQUENCY=1', 'RF']
        step += ['*RESTART, WRITE, FREQUENCY=1', '*END STEP']
        steps.append('\n'.join(step) + '\n')
    stem = out / out.name
    stem.with_suffix('.inp').write_text(header + ''.join(steps))
    (out / 'header.inp').write_text(header)
    for (j, step) in enumerate(steps):
        (out / f'stage_{j + 1:03d}.inp').write_text(step)
    np.savez(out / 'geometry.npz', nodes=vertices, object_nodes=xyz, object_elements=cells, skin_elements=np.asarray(skin['elements']) if skin is not None else np.empty((0, 6), int))
    metadata = dict(version=VERSION, pad=asdict(pad), pad_law='compressible_neo_hooke_from_declared_E_nu', material=material, motion=motion, criterion=criterion, limit=limit, shell=shell, skin=None if skin is None else dict(material=skin['material'], thickness_m=skin['thickness_m'], expected_fields=dict(element_ids=skin_ids, points_per_element=9)), secondary_criterion=secondary_criterion, secondary_limit=secondary_limit, thickness_m=thickness_m, directions=directions.tolist(), backs=backs, expected_object_fields=dict(element_ids=list(range(1, len(cells) + 1)), points_per_element=9 if shell else 4), object_nodes=list(range(1, len(xyz) + 1)), centering_nodes=center_nodes, diagnostic_symmetry_planes=list(diagnostic_symmetry_planes), fixture_nodes=sorted(fixture_nodes), centering_N_m=centering_N_m, max_increment_s=max_increment_s, constant_stages_single_increment=constant_stages, contact_penalty_Pa_m=contact_penalty_Pa_m, friction=friction, contact_slave=contact_slave, contact_type=contact_type, pad_contact_surface=pad_contact_surface, normal_control='force' if force_history is not None else 'displacement', force_anchors=force_anchors, force_refs=force_refs, source=source, requested_end_s=float(t[-1]), input_sha256=hashlib.sha256(stem.with_suffix('.inp').read_bytes()).hexdigest(), code_sha256=SOURCE_SHA256, deck_writer_sha256=SOURCE_SHA256)
    (out / 'case.json').write_text(json.dumps(metadata, indent=2))
    return metadata

def _node_complete(block, nodes):
    b = np.asarray(block, float)
    return bool(b.shape == (len(nodes), 4) and np.isfinite(b).all() and np.array_equal(np.sort(b[:, 0]), sorted(nodes)))

def assess_rows(rows, *, completed):
    damage = None
    valid_prefix = True
    for row in rows:
        valid_prefix &= row['valid']
        if valid_prefix and row['indicator_ratio'] >= 1:
            damage = row
            break
    if damage is not None:
        return dict(status='damaged', damaged=True, reason='first_valid_history_damage', first_damage_time_s=damage['t'])
    if completed and rows and all((r['valid'] for r in rows)):
        return dict(status='safe', damaged=False, reason='complete_declared_history_below_limit')
    return dict(status='unknown', damaged=None, reason='incomplete_or_invalid_history')

def normal_force_quality(reaction, target, closure, stiffness):
    target = np.asarray(target)
    support = stiffness * np.asarray(closure)
    discrepancy = abs(np.asarray(reaction) - (target - support))
    tolerance = 1e-06 + 0.05 * target
    return dict(normal_force_equilibrium_error_N=discrepancy.tolist(), normal_force_equilibrium_valid=bool(np.all(discrepancy <= tolerance)), normal_reference_support_force_N=abs(support).tolist(), normal_reference_support_valid=bool(np.all(abs(support) <= tolerance)))

def initialization_evidence(rows, initialization):
    end = initialization['end_time_s']
    tolerance = 1e-07 * max(1.0, end)
    initial = [r for r in rows if r['t'] <= end + tolerance]
    for row in initial:
        if not row['valid']:
            return dict(status='invalid', reason='incomplete_reference_initialization')
        reactions = np.asarray(row['jaw_reaction_N'])
        if np.linalg.norm(reactions, axis=1).max() > 1e-06 or row['peeq'] > 0.0:
            return dict(status='invalid', reason='reference_initialization_is_not_unloaded')
    if initial and abs(initial[-1]['t'] - end) <= tolerance:
        return dict(status='verified', end_time_s=end)
    if any((r['t'] > end + tolerance for r in rows)):
        return dict(status='invalid', reason='reference_initialization_output_missing')
    return dict(status='pending', end_time_s=end)

def analyze(directory, returncode, *, result_name='result.json', prefix_rows=(), force_resolution_N=0.0):
    out = Path(directory)
    stem = out / out.name
    case = json.loads((out / 'case.json').read_text())
    dat = str(stem) + '.dat'
    setup_errors = contact_setup_errors(case, stem.with_suffix('.solver.log'))
    force_control = case.get('normal_control') == 'force'
    streams = {}
    has_plastic = bool(case['material'].get('plastic'))
    fields = [('stress', 'stresses', 'EOBJ', 8), ('center', 'displacements', 'NCENTER', 4), ('object_u', 'displacements', 'NOBJECT', 4)]
    if has_plastic:
        fields.append(('peeq', 'equivalent plastic strain', 'EOBJ', 3))
    if case.get('skin'):
        fields.append(('skin', 'stresses', 'ESKIN', 8))
    geometry = np.load(out / 'geometry.npz') if case.get('skin') or case['criterion'] == 'paperboard_yield' or 'plastic_volume' in (case['criterion'], case.get('secondary_criterion')) else None
    if case.get('fixture_nodes'):
        fields.append(('fixture', 'forces', 'NFIXTURE', 4))
    for k in range(2):
        for (variable, field) in [('forces', 'rf'), ('displacements', 'u')]:
            nset = f'NANCHOR{k}' if force_control and field == 'rf' else f'NBACK{k}'
            fields.append((f'{field}{k}', variable, nset, 4))
        if force_control:
            fields.append((f'q{k}', 'displacements', f'NFORCE{k}', 4))
    streams = run.blocks_with_time_multi(dat, fields) if Path(dat).exists() else {name: {} for (name, _, _, _) in fields}
    (t, _, rotations) = motion_arrays(case['motion'])
    rows = list(prefix_rows)
    for time in sorted(set().union(*(set(s) for s in streams.values()))):
        good = all((time in s for s in streams.values()))
        if good:
            good &= all((run.complete_field(streams[key][time], case['expected_object_fields']) for key in (('stress', 'peeq') if has_plastic else ('stress',))))
            if case.get('skin'):
                good &= run.complete_field(streams['skin'][time], case['skin']['expected_fields'])
            good &= _node_complete(streams['center'][time], case['centering_nodes'])
            good &= _node_complete(streams['object_u'][time], case['object_nodes'])
            if case.get('fixture_nodes'):
                good &= _node_complete(streams['fixture'][time], case['fixture_nodes'])
            good &= all((_node_complete(streams[f'{v}{k}'][time], case['force_anchors'][k] if force_control and v == 'rf' else case['backs'][k]) for k in range(2) for v in ('rf', 'u')))
            if force_control:
                good &= all((_node_complete(streams[f'q{k}'][time], [case['force_refs'][k]]) for k in range(2)))
        if not good:
            rows.append(dict(t=time, valid=False, indicator_ratio=None, reason='missing_or_incomplete_requested_output'))
            continue
        stress = np.asarray(streams['stress'][time])
        rf = [np.asarray(streams[f'rf{k}'][time])[:, 1:].sum(0) for k in range(2)]
        center_u = np.asarray(streams['center'][time])[:, 1:]
        support = -case['centering_N_m'] * center_u
        fixture_reaction = np.asarray(streams['fixture'][time])[:, 1:].sum(0) if case.get('fixture_nodes') else np.zeros(3)
        j = max(1, min(int(np.searchsorted(t, time)), len(t) - 1))
        fraction = (time - t[j - 1]) / (t[j] - t[j - 1])
        rotation = (1 - fraction) * rotations[j - 1, 0] + fraction * rotations[j, 0]
        direction = rotation @ np.asarray(case['directions'][0])
        direction /= np.linalg.norm(direction)
        quality = reaction_quality(rf[0], rf[1], support, direction, absolute_N=REACTION_ABSOLUTE_N, physical_fixture=fixture_reaction)
        pmax = float(np.asarray(streams['peeq'][time])[:, 2].max()) if has_plastic else 0.0
        inv = shell_surface.invariants(stress[:, 2:])
        sf = shell_surface.reconstruct(stress)['surface'] if case['shell'] and pmax == 0 else None
        secondary_ratio = 0.0
        if case['criterion'] == 'peeq':
            value = pmax
        elif case['criterion'] == 'plastic_volume':
            value = _plastic_volume_fraction(np.asarray(streams['peeq'][time]), geometry)
        elif case['criterion'] == 's1':
            value = max(0.0, (sf or inv)['s1_max'])
        elif case['criterion'] == 's3':
            value = max(0.0, -inv['s3_min'])
        elif case['criterion'] == 'shell_yield':
            value = case['limit'] if pmax > 0 else sf['vm_max']
        elif case['criterion'] == 'skin_s1':
            value = max(0.0, shell_surface.reconstruct(np.asarray(streams['skin'][time]))['surface']['s1_max'])
        else:
            from . import reference_fields
            displacement = np.asarray(streams['object_u'][time])[:, 1:]
            if case['criterion'] == 'skin_membrane':
                try:
                    value = reference_fields.membrane_resultant(np.asarray(streams['skin'][time]), geometry['object_nodes'], displacement, geometry['skin_elements'], case['skin']['thickness_m'], case['skin']['material']['nu'])
                except reference_fields.ReferenceDomainError as exc:
                    rows.append(dict(t=time, valid=False, indicator_ratio=None, reason=str(exc)))
                    continue
            elif case['criterion'] == 'paperboard_yield':
                value = reference_fields.paperboard_surface_ratio(stress, geometry['object_nodes'], displacement, geometry['object_elements'], case['material']['orientation_axis'])
            else:
                raise ValueError('unsupported stored criterion')
        if case.get('secondary_criterion'):
            secondary = _plastic_volume_fraction(np.asarray(streams['peeq'][time]), geometry) if case['secondary_criterion'] == 'plastic_volume' else {'peeq': pmax, 's1': max(0.0, inv['s1_max']), 's3': max(0.0, -inv['s3_min'])}[case['secondary_criterion']]
            secondary_ratio = secondary / case['secondary_limit']
        row = dict(t=time, field_complete=True, **quality, indicator_ratio=max(value / case['limit'], secondary_ratio), primary_indicator_ratio=value / case['limit'], secondary_indicator_ratio=secondary_ratio, peeq=pmax, ip_stress=inv, elastic_surface_stress=sf, F=min((abs(float(f @ direction)) for f in rf)), jaw_reaction_N=[f.tolist() for f in rf], fixture_reaction_N=fixture_reaction.tolist(), jaw_mean_displacement_m=[np.asarray(streams[f'u{k}'][time])[:, 1:].mean(0).tolist() for k in range(2)], maximum_object_displacement_m=float(np.linalg.norm(np.asarray(streams['object_u'][time])[:, 1:], axis=1).max()), centering_energy_J=float(0.5 * case['centering_N_m'] * (center_u ** 2).sum()))
        if force_control:
            target = np.array([np.interp(time, t, np.asarray(case['motion']['normal_forces_N'])[:, k]) for k in range(2)])
            closure = np.array([streams[f'q{k}'][time][0][1] for k in range(2)])
            force_projection = np.array([f @ np.asarray(d) for (f, d) in zip(rf, case['directions'])])
            force_quality = normal_force_quality(force_projection, target, closure, case['centering_N_m'])
            valid_force = force_quality['normal_force_equilibrium_valid'] and force_quality['normal_reference_support_valid']
            row.update(normal_closure_m=closure.tolist(), requested_normal_force_N=target.tolist(), **force_quality, valid=row['valid'] and valid_force)
            if not force_quality['normal_force_equilibrium_valid']:
                row['reason'] = 'prescribed_normal_force_not_recovered_in_reaction'
            elif not force_quality['normal_reference_support_valid']:
                row['reason'] = 'normal_reference_carries_excessive_load'
        domain = case.get('source', {}).get('material_domain', {}).get('max_peeq')
        if domain is not None and pmax > domain:
            row.update(valid=False, reason='constitutive_table_domain_exceeded')
        rows.append(row)
    if rows and np.any(np.diff([r['t'] for r in rows]) <= 0):
        raise ValueError('history and restart prefix must form one strictly increasing time series')
    if setup_errors:
        for row in rows:
            row.update(valid=False, reason='invalid_contact_constraint_setup')
    initialization = case['source'].get('reference_initialization')
    initialization_result = None
    if initialization:
        initialization_result = initialization_evidence(rows, initialization)
        if initialization_result['status'] == 'invalid' and rows:
            rows[0].update(valid=False, reason=initialization_result['reason'])
    sta = stem.with_suffix('.sta')
    records = [line.split() for line in sta.read_text().splitlines() if line.strip() and line.split()[0].isdigit()] if sta.exists() else []
    tolerance = 1e-07 * max(1, case['requested_end_s'])
    completed = bool(returncode == 0 and records and rows and (abs(float(records[-1][4]) - case['requested_end_s']) <= tolerance) and (abs(rows[-1]['t'] - case['requested_end_s']) <= tolerance))
    result = dict(version=VERSION, returncode=returncode, requested_path_completed=completed, **assess_rows(rows, completed=completed), rows=rows, source=case['source'], analysis_code_sha256=SOURCE_SHA256, case_sha256=hashlib.sha256((out / 'case.json').read_bytes()).hexdigest(), input_sha256=case['input_sha256'], contact_setup_errors=setup_errors)
    result['force_resolution_N'] = force_resolution_N
    if initialization_result is not None:
        result['reference_initialization'] = initialization_result
    if rows:
        result['force_only_bounds'] = threshold_bounds([r.get('F', np.nan) for r in rows], [r['indicator_ratio'] if r['indicator_ratio'] is not None else np.nan for r in rows], 1, valid=[r['valid'] for r in rows], completed=completed, force_resolution_N=force_resolution_N).to_dict()
    (out / result_name).write_text(json.dumps(result, indent=2, allow_nan=False))
    return result

def contact_setup_errors(case, solver_log):
    errors = []
    if case.get('contact_type') == 'MORTAR' and case.get('contact_slave') == 'object':
        errors.append('full_object_is_slave_in_both_mortar_pairs')
    log = Path(solver_log)
    if log.exists() and 'belongs to both slave surface' in log.read_text():
        errors.append('solver_removed_duplicate_slave_lagrange_multipliers')
    return errors

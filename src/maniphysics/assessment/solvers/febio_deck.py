import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as X
import numpy as np
ROOT = Path(os.environ.get('MPB_RUNTIME_ROOT', 'runtime')).resolve()
PLUGIN = ROOT / 'plugin/compare.so'
FACES = {1: [2, 1, 0, 5, 4, 6], 2: [0, 1, 3, 4, 8, 7], 3: [1, 2, 3, 5, 9, 8], 4: [2, 0, 3, 6, 7, 9]}

def sub(parent, tag, text=None, **attrs):
    a = X.SubElement(parent, tag, {k: str(v) for (k, v) in attrs.items()})
    if text is not None:
        a.text = str(text)
    return a

def sections(path):
    result = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('**'):
            continue
        if line.startswith('*'):
            s = line.split(',')
            h = s[0].upper()
            attrs = {}
            for item in s[1:]:
                if '=' in item:
                    (k, v) = item.split('=', 1)
                    attrs[k.strip().upper()] = v.strip()
            result.append((h, attrs, []))
        else:
            result[-1][2].append(line)
    return result

def translate(source, out, *, penalty=None, augmented=False, steps=20, domain='elastic-solid', check_jacobians=False, search_radius=0.01, material_writer=None):
    out.mkdir(parents=True, exist_ok=True)
    inp = source / (source.name + '.inp')
    ss = sections(inp)
    case = json.loads((source / 'case.json').read_text())
    assert not case['shell']
    assert not any((h == '*CLOAD' for (h, a, l) in ss)), 'only prescribed displacement source decks'
    nodes = {}
    els = {}
    groups = {}
    surfaces = {}
    amps = {}
    bc = []
    pairs = []
    for (h, a, lines) in ss:
        if h == '*NODE':
            for l in lines:
                v = l.split(',')
                nodes[int(v[0])] = list(map(float, v[1:]))
        elif h == '*ELEMENT' and a['TYPE'] in ('C3D10', 'S6'):
            ids = []
            for l in lines:
                v = list(map(int, l.split(',')))
                els[v[0]] = v[1:]
                ids.append(v[0])
            groups[a['ELSET']] = ids
        elif h == '*SURFACE':
            surfaces[a['NAME']] = [(int(l.split(',')[0]), int(l.split(',')[1].strip()[1:])) for l in lines]
        elif h == '*AMPLITUDE':
            amps[a['NAME']] = tuple((tuple(map(float, l.split(','))) for l in lines))
        elif h == '*CONTACT PAIR':
            pairs.extend((tuple((s.strip() for s in l.split(','))) for l in lines))
        elif h == '*BOUNDARY':
            for l in lines:
                v = l.split(',')
                (n, lo, hi) = map(int, v[:3])
                val = float(v[3])
                curve = amps[a['AMPLITUDE']] if 'AMPLITUDE' in a else ((0.0, val), (case['requested_end_s'], val))
                if 'AMPLITUDE' in a:
                    curve = tuple(((t, u * val) for (t, u) in curve))
                for d in range(lo, hi + 1):
                    bc.append((n, d, curve))
    ground = []
    tethers = []
    for n in case['centering_nodes']:
        g = max(nodes) + 1
        nodes[g] = nodes[n].copy()
        ground.append(g)
        tethers.append((n, g))
        for d in (1, 2, 3):
            bc.append((g, d, ((0.0, 0.0), (case['requested_end_s'], 0.0))))
    spec = X.Element('febio_spec', version='4.0')
    sub(spec, 'Module', type='solid')
    ctrl = sub(spec, 'Control')
    sub(ctrl, 'analysis', 'STATIC')
    for (k, v) in dict(time_steps=steps, step_size=case['requested_end_s'] / steps, plot_level='PLOT_NEVER', output_level='OUTPUT_MAJOR_ITRS').items():
        sub(ctrl, k, v)
    solver = sub(ctrl, 'solver', type='solid')
    for (k, v) in dict(symmetric_stiffness='non-symmetric', dtol=1e-05, etol=1e-06, rtol=0, max_refs=30).items():
        sub(solver, k, v)
    if check_jacobians:
        sub(solver, 'ls_check_jacobians', 1)
    sub(sub(solver, 'qn_method', type='BFGS'), 'max_ups', 10)
    stepper = sub(ctrl, 'time_stepper', type='default')
    for (k, v) in dict(dtmin=1e-07, dtmax=case['requested_end_s'] / steps, max_retries=8, opt_iter=10).items():
        sub(stepper, k, v)
    material = sub(spec, 'Material')
    m = sub(material, 'material', id=1, name='object', type='comparison native tracked J2')
    if material_writer is None:
        pl = np.asarray(case['material']['plastic'], float)
        assert len(pl) == 2 and pl[0, 1] == 0, 'Native FEBio J2 mapping supports only a linear hardening branch'
        slope = (pl[1, 0] - pl[0, 0]) / (pl[1, 1] - pl[0, 1])
        sub(m, 'E', case['material']['E'])
        sub(m, 'v', case['material']['nu'])
        sub(m, 'Y', pl[0, 0])
        sub(m, 'H', 2 * slope / 3)
    else:
        material_mapping = material_writer(m, case['material'])
    if case.get('skin'):
        sk = case['skin']['material']
        ms = sub(material, 'material', id=3, name='skin', type='isotropic elastic')
        sub(ms, 'E', sk['E'])
        sub(ms, 'v', sk['nu'])
    p = case['pad']
    m = sub(material, 'material', id=2, name='pad', type='incomp neo-Hookean')
    sub(m, 'G', p['E_Pa'] / (2 * (1 + p['nu'])))
    sub(m, 'k', p['E_Pa'] / (3 * (1 - 2 * p['nu'])))
    sub(m, 'pressure_model', 2)
    material[:] = sorted(material, key=lambda m: int(m.attrib['id']))
    mesh = sub(spec, 'Mesh')
    nn = sub(mesh, 'Nodes', name='all')
    for (n, xyz) in nodes.items():
        sub(nn, 'node', ','.join((format(v, '.12g') for v in xyz)), id=n)
    for (name, ids) in groups.items():
        ee = sub(mesh, 'Elements', type='tri6' if name == 'ESKIN' else 'tet10', name=name)
        for i in ids:
            sub(ee, 'elem', ','.join(map(str, els[i])), id=i)
    for (name, faces) in surfaces.items():
        s = sub(mesh, 'Surface', name=name)
        for (j, (i, f)) in enumerate(faces, 1):
            face = FACES[f] if len(els[i]) == 10 else [0, 2, 1, 5, 4, 3] if f == 1 else [0, 1, 2, 3, 4, 5]
            sub(s, 'tri6', ','.join((str(els[i][v]) for v in face)), id=j)
    for k in range(2):
        pair = sub(mesh, 'SurfacePair', name=f'pair{k}')
        (primary, secondary) = pairs[k]
        sub(pair, 'primary', primary)
        sub(pair, 'secondary', secondary)
    for (name, ids) in [('back0', case['backs'][0]), ('back1', case['backs'][1]), ('center', case['centering_nodes']), ('ground', ground), ('object', case['object_nodes'])]:
        sub(mesh, 'NodeSet', ','.join(map(str, ids)), name=name)
    sub(mesh, 'ElementSet', ','.join(map(str, groups['EOBJ'])), name='object')
    ds = sub(mesh, 'DiscreteSet', name='tethers')
    for (i, pair) in enumerate(tethers, 1):
        sub(ds, 'delem', ','.join(map(str, pair)), id=i)
    grouped = {}
    for (n, d, c) in bc:
        grouped.setdefault((d, c), set()).add(n)
    assigned = {}
    for (n, d, c) in bc:
        assert (n, d) not in assigned or assigned[n, d] == c
        assigned[n, d] = c
    domains = sub(spec, 'MeshDomains')
    for name in groups:
        if name == 'ESKIN':
            sd = sub(domains, 'ShellDomain', name=name, mat='skin', type='elastic-shell', elem_type='TRI6G21')
            sub(sd, 'shell_thickness', case['skin']['thickness_m'])
            sub(sd, 'shell_normal_nodal', 0)
        else:
            sub(domains, 'SolidDomain', name=name, mat='object' if name == 'EOBJ' else 'pad', elem_type='TET10G4', type=domain)
    boundary = sub(spec, 'Boundary')
    curves = {}
    for (k, ((d, c), ids)) in enumerate(grouped.items()):
        name = f'bc{k}'
        sub(mesh, 'NodeSet', ','.join(map(str, sorted(ids))), name=name)
        curves.setdefault(c, len(curves) + 1)
        b = sub(boundary, 'bc', name=name, type='prescribed displacement', node_set=name)
        sub(b, 'dof', 'xyz'[d - 1])
        sub(b, 'value', 1.0, lc=curves[c])
        sub(b, 'relative', 0)
    discrete = sub(spec, 'Discrete')
    dm = sub(discrete, 'discrete_material', id=1, type='basis Cartesian tether')
    sub(dm, 'k', case['centering_N_m'])
    sub(discrete, 'discrete', dmat=1, discrete_set='tethers')
    contact = sub(spec, 'Contact')
    for k in range(2):
        c = sub(contact, 'contact', type='sliding-elastic', surface_pair=f'pair{k}')
        for (name, val) in dict(laugon=int(augmented), penalty=penalty or case['contact_penalty_Pa_m'], auto_penalty=0, two_pass=0, search_tol=0.01, search_radius=search_radius, symmetric_stiffness=0, fric_coeff=case['friction'], tolerance=0.01, gaptol=1e-08, minaug=0, maxaug=10).items():
            sub(c, name, val)
    ld = sub(spec, 'LoadData')
    for (c, i) in curves.items():
        lc = sub(ld, 'load_controller', id=i, type='loadcurve')
        sub(lc, 'interpolate', 'LINEAR')
        pts = sub(lc, 'points')
        for (t, u) in c:
            sub(pts, 'pt', f'{t:.12g},{u:.12g}')
    output = sub(spec, 'Output')
    logs = sub(output, 'logfile')
    for name in ('back0', 'back1', 'center', 'ground', 'object'):
        sub(logs, 'node_data', data='ux;uy;uz;Rx;Ry;Rz', node_set=name, file=name + '.txt')
    sub(logs, 'element_data', data='sx;sy;sz;sxy;syz;sxz;basis ip max vm;basis ip max s1;basis ip min J;basis ip count;comparison mean peeq', elem_set='object', file='stress.txt')
    if case.get('skin'):
        sub(mesh, 'ElementSet', ','.join(map(str, groups['ESKIN'])), name='skin')
        for i in range(21):
            sub(logs, 'element_data', data=';'.join((f'comparison s{i}_{j}' for j in range(6))), elem_set='skin', file=f'skin_ip{i}.txt')
    X.indent(spec)
    X.ElementTree(spec).write(out / 'model.feb', encoding='utf-8', xml_declaration=True)
    meta = dict(source=str(source), source_input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(), input_sha256=hashlib.sha256((out / 'model.feb').read_bytes()).hexdigest(), plugin_sha256=hashlib.sha256(PLUGIN.read_bytes()).hexdigest(), code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), domain=domain, material_mapping=material_mapping if material_writer is not None else None, steps=steps, augmented=augmented, penalty=penalty or case['contact_penalty_Pa_m'], source_case=case)
    meta['numerics'] = dict(ls_check_jacobians=check_jacobians, search_radius_m=search_radius)
    (out / 'input_audit.json').write_text(json.dumps(meta, indent=2))
    return meta

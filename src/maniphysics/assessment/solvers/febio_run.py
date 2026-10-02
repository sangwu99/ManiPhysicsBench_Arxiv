import hashlib
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as X
import numpy as np
ROOT = Path(os.environ.get('MPB_RUNTIME_ROOT', 'runtime')).resolve()
from . import febio_native as native
from .febio_log import read_log
from maniphysics.assessment.core.quality import reaction_quality
OUT = ROOT

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def set_value(parent, name, value):
    node = parent.find(name)
    if node is None:
        node = X.SubElement(parent, name)
    node.text = str(value)
    return node

def prepare(args):
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=False)
    original = json.loads((args.source / 'input_audit.json').read_text())
    tree = X.parse(args.source / 'model.feb')
    root = tree.getroot()
    case = original['source_case']

    ctrl = root.find('Control')
    duration = args.ramp_duration
    end = 1 + args.fraction * duration
    set_value(ctrl, 'time_steps', 1)
    set_value(ctrl, 'step_size', end)
    stepper = ctrl.find('time_stepper')
    dtmax = stepper.find('dtmax')
    ld = root.find('LoadData')
    if 'lc' in dtmax.attrib:
        curve = ld.find(f"load_controller[@id='{dtmax.attrib['lc']}']")
    else:
        cid = max((int(c.attrib['id']) for c in ld)) + 1
        dtmax.set('lc', str(cid))
        curve = X.SubElement(ld, 'load_controller', id=str(cid), type='loadcurve')
        X.SubElement(curve, 'interpolate').text = 'STEP'
        X.SubElement(curve, 'points')
    pts = curve.find('points')
    pts.clear()
    times = [(0, 1), (1, args.dt * duration), (1 + duration, args.dt * duration)]
    if args.initial_step is not None:
        times = [(0, 1), (1, args.initial_step * duration), (1 + args.initial_step * duration, args.dt * duration), (1 + duration, args.dt * duration)]
    for (t, value) in times:
        X.SubElement(pts, 'pt').text = f'{t},{value}'
    dtmax.text = '0'
    if args.aggressive:
        set_value(stepper, 'aggressiveness', 1)
        set_value(stepper, 'cutback', 0.5)
    if args.max_retries is not None:
        set_value(stepper, 'max_retries', args.max_retries)
    if args.opt_iter is not None:
        set_value(stepper, 'opt_iter', args.opt_iter)
    mode = args.mode
    if mode == 'no_contact':
        root.remove(root.find('Contact'))
    if mode == 'elastic_object':
        mat = root.find("Material/material[@id='1']")
        mat.clear()
        mat.attrib.update(id='1', name='object', type='natural neo-Hookean')
        set_value(mat, 'E', case['material']['E'])
        set_value(mat, 'v', case['material']['nu'])
        for el in root.findall('Output/logfile/element_data'):
            el.set('data', el.attrib['data'].replace(native.Q, 'sx'))
    if mode == 'frictionless':
        for c in root.findall('Contact/contact'):
            set_value(c, 'fric_coeff', 0)
    if mode == 'strong_tether':
        root.find('Discrete/discrete_material/k').text = str(args.tether)
    if mode == 'fixed_object':
        for dof in 'xyz':
            b = X.SubElement(root.find('Boundary'), 'bc', type='prescribed displacement', node_set='object')
            set_value(b, 'dof', dof)
            set_value(b, 'value', 0)
            set_value(b, 'relative', 0)
    if mode == 'small_search':
        for c in root.findall('Contact/contact'):
            set_value(c, 'search_radius', args.search_radius)
    if mode == 'node_on_facet':
        supported = {'laugon', 'tolerance', 'penalty', 'auto_penalty', 'two_pass', 'gaptol', 'fric_coeff', 'fric_penalty', 'minaug', 'maxaug', 'search_tol', 'ktmult', 'knmult', 'node_reloc', 'seg_up', 'search_radius', 'update_penalty'}
        for c in root.findall('Contact/contact'):
            c.set('type', 'sliding-node-on-facet')
            for el in list(c):
                if el.tag not in supported:
                    c.remove(el)
            set_value(c, 'fric_penalty', args.friction_penalty)
    if args.solver:
        q = root.find('Control/solver/qn_method')
        q.clear()
        q.set('type', args.solver)
        if args.solver in ('BFGS', 'Broyden'):
            set_value(q, 'max_ups', 10)
    if args.penalty is not None:
        for c in root.findall('Contact/contact'):
            set_value(c, 'penalty', args.penalty)
    if args.maxaug is not None:
        for c in root.findall('Contact/contact'):
            set_value(c, 'maxaug', args.maxaug)
    if args.gaptol is not None:
        for c in root.findall('Contact/contact'):
            set_value(c, 'gaptol', args.gaptol)
    if args.contact_offset:
        for c in root.findall('Contact/contact'):
            set_value(c, 'offset', args.contact_offset)
    if args.node_reloc:
        for c in root.findall('Contact/contact'):
            set_value(c, 'node_reloc', 1)
    if args.two_pass:
        for c in root.findall('Contact/contact'):
            set_value(c, 'two_pass', 1)
    if args.seg_up is not None:
        for c in root.findall('Contact/contact'):
            set_value(c, 'seg_up', args.seg_up)
    if args.lsmin is not None:
        set_value(root.find('Control/solver'), 'lsmin', args.lsmin)
        set_value(root.find('Control/solver'), 'rtol', 1e-06)
    if args.surface_rule:
        for domain in root.findall('MeshDomains/SolidDomain'):
            domain.set('elem_type', 'TET10G4_S' + args.surface_rule[-1])
    if args.pad_three_field:
        for domain in root.findall('MeshDomains/SolidDomain'):
            if domain.attrib['name'].startswith('EPAD'):
                domain.set('type', 'three-field-solid')
    if duration != 1:
        for lc in ld:
            if lc.attrib['id'] == dtmax.attrib['lc']:
                continue
            for point in lc.findall('points/pt'):
                (t, value) = map(float, point.text.split(','))
                point.text = f'{(t if t <= 1 else 1 + (t - 1) * duration):.16g},{value:.16g}'
    if args.dynamic:
        assert args.length_scale == 1
        set_value(ctrl, 'analysis', 'DYNAMIC')
        set_value(root.find('Control/solver'), 'rhoi', 0)
        for mat in root.findall('Material/material'):
            density = args.object_density if mat.attrib.get('name') == 'object' else args.pad_density
            set_value(mat, 'density', density if density is not None else args.density)
    if args.rtol is not None:
        set_value(root.find('Control/solver'), 'rtol', args.rtol)
    if args.dtol is not None:
        set_value(root.find('Control/solver'), 'dtol', args.dtol)
    if args.etol is not None:
        set_value(root.find('Control/solver'), 'etol', args.etol)
    scale = args.length_scale
    if scale != 1:
        for n in root.findall('Mesh/Nodes/node'):
            n.text = ','.join((format(float(v) * scale, '.16g') for v in n.text.split(',')))
        for mat in root.findall('Material/material'):
            for key in ('E', 'G', 'k'):
                for el in mat.iter(key):
                    el.text = format(float(el.text) / (scale * scale), '.16g')
            for point in mat.findall('flow_curve/plastic_response/points/point'):
                (strain, stress) = map(float, point.text.split(','))
                point.text = f'{strain:.16g},{stress / (scale * scale):.16g}'
        for el in root.findall('Discrete/discrete_material/k'):
            el.text = format(float(el.text) / scale, '.16g')
        for lc in ld:
            if lc.attrib['id'] == dtmax.attrib['lc']:
                continue
            for point in lc.findall('points/pt'):
                (t, value) = map(float, point.text.split(','))
                point.text = f'{t:.16g},{value * scale:.16g}'
        for c in root.findall('Contact/contact'):
            for name in ('search_radius', 'gaptol'):
                e = c.find(name)
                e.text = format(float(e.text) * scale, '.16g')
            if c.find('fric_penalty') is not None:
                e = c.find('fric_penalty')
                e.text = format(float(e.text) / scale ** 3, '.16g')
    mesh = root.find('Mesh')
    logs = root.find('Output/logfile')
    for e in root.findall('Mesh/Elements'):
        if args.dynamic:
            ename = 'energy_' + e.attrib['name']
            X.SubElement(mesh, 'ElementSet', name=ename).text = ','.join((x.attrib['id'] for x in e))
            X.SubElement(logs, 'element_data', elem_set=ename, file=ename + '.txt', data='basis element strain energy;basis element kinetic energy')
        if e.attrib['name'] == 'EOBJ':
            continue
        name = 'diag_' + e.attrib['name']
        X.SubElement(mesh, 'ElementSet', name=name).text = ','.join((x.attrib['id'] for x in e))
        X.SubElement(logs, 'element_data', elem_set=name, file=name + '.txt', data='sx;sy;sz;sxy;syz;sxz;basis ip min J')
    if mode != 'no_contact':
        for k in (0, 1):
            X.SubElement(logs, 'surface_data', surface=f'SPAD{k}', file=f'contact{k}.txt', data='contact area;max contact gap')
    plugin = native.PLUGIN
    X.indent(root)
    tree.write(out / 'model.feb', encoding='utf-8', xml_declaration=True)
    meta = dict(source=str(args.source), source_deck_sha256=sha(args.source / 'model.feb'), source_case=case, original_metadata=original, mode=mode, end_time=end, dt=args.dt, options=vars(args) | {'source': str(args.source)}, native_units='N,m,Pa' if scale == 1 else 'N,mm,MPa', length_scale_from_m=scale, ramp_duration_s=duration, dynamic=args.dynamic, input_sha256=sha(out / 'model.feb'), script_sha256=sha(Path(__file__)), plugin=str(plugin), plugin_sha256=sha(plugin), source_model_preserved=mode in ('baseline', 'small_search', 'node_on_facet'))
    (out / 'input_audit.json').write_text(json.dumps(meta, indent=2))
    return out

def analyze(out, execution):
    m = json.loads((out / 'input_audit.json').read_text())
    case = m['source_case']
    init = case['source'].get('reference_initialization') or {}
    load_start = float(init.get('end_time_s', 0.0))
    scale = m.get('length_scale_from_m', 1)
    files = ['stress', 'back0', 'back1', 'center']
    streams = {k: dict(read_log(out / (k + '.txt'))) for k in files if (out / (k + '.txt')).exists()}
    contacts = {k: dict(read_log(out / f'contact{k}.txt')) for k in (0, 1) if (out / f'contact{k}.txt').exists()}
    energy = {p.stem: dict(read_log(p)) for p in out.glob('energy_*.txt')}
    rows = []
    if len(streams) == len(files):
        source = Path(m['original_metadata']['source'])
        if not source.is_absolute():
            source = ROOT / source
        geom = np.load(source / 'geometry.npz')
        xyz = geom['nodes'][geom['object_elements'][:, :4]]
        vol = np.abs(np.einsum('ij,ij->i', xyz[:, 1] - xyz[:, 0], np.cross(xyz[:, 2] - xyz[:, 0], xyz[:, 3] - xyz[:, 0]))) / 6
        evidence = case['source']['material_evidence']
        allowances = evidence.get('b_axis_declared_allowance')
        if isinstance(allowances, dict) and case['source']['model_id'] in allowances:
            allowance = allowances[case['source']['model_id']]
        elif case.get('criterion') == 'plastic_volume' and case.get('limit') is not None:
            allowance = case['limit']
        elif case.get('secondary_criterion') == 'plastic_volume':
            allowance = case['secondary_limit']
        else:
            raise KeyError('plastic-volume allowance is absent from both material evidence and case.limit')
        for (t, st) in streams['stress'].items():
            if not all((t in streams[k] for k in files)):
                rows.append(dict(t=t, valid=False, min_J=0, reason='incomplete_nodal_output'))
                continue
            st = st[np.argsort(st[:, 0])]
            rf = [-streams[f'back{k}'][t][:, 4:7].sum(0) for k in (0, 1)]
            k = m['options']['tether'] if m['mode'] == 'strong_tether' else case['centering_N_m']
            support = -k * streams['center'][t][:, 1:4] / scale
            quality = reaction_quality(*rf, support, case['directions'][0], absolute_N=0.001)
            row = dict(t=t, closing_fraction=(t - load_start) / m.get('ramp_duration_s', 1), F=min((abs(f @ np.array(case['directions'][0])) for f in rf)), min_J=float(st[:, 9].min()), max_vm_Pa=float(st[:, 7].max()) * scale ** 2, center_displacement_max_m=float(np.linalg.norm(streams['center'][t][:, 1:4], axis=1).max()) / scale, jaw_reaction_N=[f.tolist() for f in rf], **quality)
            if m['mode'] != 'elastic_object':
                q = st[:, -1]
                phi = float(vol[q >= 0.002].sum() / vol.sum())
                row.update(q_max=float(q.max()), plastic_volume_fraction=phi, indicator_ratio=phi / allowance)
            if case.get('skin'):
                from .febio_skin import membrane_ratio
                primary = membrane_ratio(out, t, case, geom)
                row.update(primary_indicator_ratio=primary, secondary_indicator_ratio=phi / allowance, indicator_ratio=max(primary, phi / allowance))
            row['contact'] = {str(k): contacts[k][t].tolist() for k in contacts if t in contacts[k]}
            if len(contacts) == 2 and all((t in contacts[k] for k in contacts)):
                row['bilateral_contact'] = bool(all((float(contacts[k][t][:, 1].sum()) > 0 for k in (0, 1))))
                row['max_contact_gap_m'] = max((float(contacts[k][t][:, -1].max()) for k in contacts)) / scale
                gap_limit = m['options'].get('gaptol') or 1e-06
                gap_comparison_slack = max(1e-09, 0.001 * gap_limit)
                row['contact_gap_limit_m'] = gap_limit
                row['contact_gap_comparison_slack_m'] = gap_comparison_slack
                row['contact_gap_valid'] = row['max_contact_gap_m'] <= gap_limit + gap_comparison_slack
                row['reaction_quality_valid'] = row['valid']
                row['valid'] = bool(row['valid'] and row['contact_gap_valid'])
            else:
                row.update(valid=False, reason='incomplete_contact_output')
            if energy and all((t in e for e in energy.values())):
                ie = sum((float(e[t][:, 1].sum()) for e in energy.values()))
                ke = sum((float(e[t][:, 2].sum()) for e in energy.values()))
                row.update(strain_energy_J=ie, kinetic_energy_J=ke, energy_valid=bool(ie >= 0 and ke >= 0 and (ke <= 1e-08 + 0.05 * ie)))
                row['valid'] = bool(row['valid'] and row['energy_valid'])
            if m['dynamic'] and (len(energy) != 3 + int(bool(case.get('skin'))) or
                                 not all(t in e for e in energy.values())):
                row.update(valid=False, reason='incomplete_energy_output')
            rows.append(row)
    text = (out / 'stdout.log').read_text()
    prefix = []
    for row in rows:
        if row['t'] <= load_start:
            continue
        if not row['valid'] or row['min_J'] <= 0:
            break
        prefix.append(row)
    start = next((i for (i, r) in enumerate(rows) if r['t'] > load_start and r['valid'] and r.get('bilateral_contact', False) and (r['F'] >= 0.001)), None)
    evaluation = []
    if start is not None:
        for row in rows[start:]:
            if not row['valid'] or row['min_J'] <= 0:
                break
            evaluation.append(row)
    hit = next((i for (i, r) in enumerate(evaluation) if r.get('indicator_ratio', 0) >= 1), None)
    changed_contact_initialization = bool(m['options'].get('contact_offset') or m['options'].get('node_reloc') or m['options'].get('two_pass'))
    candidate = None if hit is None or not m['source_model_preserved'] or changed_contact_initialization else dict(force_interval_N=[evaluation[hit - 1]['F'] if hit else 0, evaluation[hit]['F']], time_interval_s=[evaluation[hit - 1]['t'] if hit else evaluation[hit]['t'], evaluation[hit]['t']])
    precontact = rows[:start] if start is not None else rows
    result = dict(execution=execution, rows=rows, completed=bool(execution['returncode'] == 0 and rows and (abs(rows[-1]['t'] - m['end_time']) < 1e-06) and ('N O R M A L   T E R M I N A T I O N' in text)), valid_prefix_rows=len(prefix), native_model_candidate=candidate, loading_start_time_s=load_start, evaluation_start_time_s=None if start is None else rows[start]['t'], evaluation_prefix_rows=len(evaluation), precontact_rows=len(precontact), precontact_max_indicator_ratio=max((r.get('indicator_ratio', 0) for r in precontact), default=0), precontact_max_center_displacement_m=max((r.get('center_displacement_max_m', 0) for r in precontact), default=0), contact_initialization_changed=changed_contact_initialization, failure_counts={k: text.count(v) for (k, v) in dict(negative_jacobians='negative jacobians detected', zero_line_step='Zero linestep size', plasticity_iterations='Plasticity iterations did not converge', max_reformations='Max nr of reformations reached', max_retries='Max. nr of retries reached').items()}, attempted_times=[float(x) for x in re.findall('beginning time step \\d+ : ([\\d.eE+\\-]+)', text)])
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    return result

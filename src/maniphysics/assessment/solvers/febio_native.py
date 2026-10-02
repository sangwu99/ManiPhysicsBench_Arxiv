import json
import os
from pathlib import Path
import xml.etree.ElementTree as X
ROOT = Path(os.environ.get('MPB_RUNTIME_ROOT', 'runtime')).resolve()
from . import febio_deck as base
OUT = ROOT
PLUGIN = ROOT / 'plugin/reactive.so'
Q = 'reactive accumulated equivalent plastic strain'
NET = 'reactive net equivalent plastic strain'
OCT = 'reactive weighted native octahedral strain'

def material(parent, E, nu, plastic, tracked=True, softening=None):
    if softening is not None and (not tracked or len(plastic) != 1 or plastic[0][1] != 0):
        raise ValueError('Yield softening requires one initial yield point and accumulated plastic strain tracking')
    parent.clear()
    parent.attrib.update(id='1', name='object', type='tracked native reactive plasticity' if tracked else 'reactive plasticity')
    base.sub(parent, 'isochoric', 1)
    base.sub(parent, 'rtol', 1e-06)
    e = base.sub(parent, 'elastic', type='natural neo-Hookean')
    base.sub(e, 'E', E)
    base.sub(e, 'v', nu)
    base.sub(parent, 'yield_criterion', type='DC von Mises stress')
    f = base.sub(parent, 'flow_curve', type='PFC user')
    p = base.sub(f, 'plastic_response', type='point')
    base.sub(p, 'interpolate', 'LINEAR')
    pts = base.sub(p, 'points')
    curve = []
    for (stress, peeq) in plastic:
        strain = peeq + stress / E
        curve.append([strain, stress])
        base.sub(pts, 'point', f'{strain:.16g},{stress:.16g}')
    mapping = dict(E_Pa=E, nu=nu, source_true_stress_vs_peeq=plastic, febio_total_log_strain_vs_true_stress=curve, base_elastic='natural neo-Hookean')
    if softening is not None:
        parent.set('type', 'history yield softening')
        base.sub(parent, 'yield0', plastic[0][0])
        base.sub(parent, 'residual', softening['residual'])
        base.sub(parent, 'softening_scale', softening['scale'])
        mapping.update(yield0_Pa=plastic[0][0], residual=softening['residual'],
                       softening_scale=softening['scale'])
    return mapping

def contact(case, tag, timeout=1800, auto_penalty=False, penalty=10000000000.0, source=None, gaptol=1e-08, native_pad=False, fast_init=False, object_primary=False, no_line_search=False, min_residual=None):
    source = Path(source)
    out = OUT / case / tag
    assert not out.exists()
    def write_material(parent, values):
        return material(parent, values['E'], values['nu'], values['plastic'],
                        softening=values.get('yield_softening'))
    m = base.translate(source, out, steps=100, penalty=penalty, augmented=True,
                       check_jacobians=True, material_writer=write_material)
    f = X.parse(out / 'model.feb')
    root = f.getroot()
    mapping = m['material_mapping']
    for el in root.findall('Output/logfile/element_data'):
        el.set('data', el.attrib['data'].replace('comparison mean peeq', Q))
    so = root.find('Control/solver')
    so.find('dtol').text = '.001'
    so.find('etol').text = '.01'
    so.find('qn_method').set('type', 'full Newton')
    so.find('qn_method').clear()
    so.find('qn_method').set('type', 'full Newton')
    if native_pad:
        for d in root.findall('MeshDomains/SolidDomain'):
            if d.attrib['mat'] == 'pad':
                d.set('type', 'three-field-solid')
        so.find('qn_method').set('type', 'BFGS')
        base.sub(so.find('qn_method'), 'max_ups', 10)
    if object_primary:
        for pair in root.findall('Mesh/SurfacePair'):
            a = pair.find('primary')
            b = pair.find('secondary')
            (a.text, b.text) = (b.text, a.text)
    if no_line_search:
        base.sub(so, 'lstol', 0)
    if min_residual is not None:
        base.sub(so, 'min_residual', min_residual)
    if fast_init:
        ctrl = root.find('Control')
        ctrl.find('time_steps').text = '2'
        ctrl.find('step_size').text = '1'
        ld = root.find('LoadData')
        lc = max((int(x.attrib['id']) for x in ld)) + 1
        curve = base.sub(ld, 'load_controller', id=lc, type='loadcurve')
        base.sub(curve, 'interpolate', 'STEP')
        pts = base.sub(curve, 'points')
        for (t, v) in [(0, 1), (1, 0.02), (2, 0.02)]:
            base.sub(pts, 'pt', f'{t},{v}')
        ctrl.find('time_stepper/dtmax').set('lc', str(lc))
        ctrl.find('time_stepper/dtmax').text = '1'
    if auto_penalty:
        for c in root.findall('Contact/contact'):
            c.find('auto_penalty').text = '1'
            c.find('penalty').text = str(penalty)
    for c in root.findall('Contact/contact'):
        c.find('gaptol').text = str(gaptol)
    X.indent(root)
    f.write(out / 'model.feb', encoding='utf-8', xml_declaration=True)
    m.update(material_mapping=mapping, contact_auto_penalty=auto_penalty, gaptol_m=gaptol, pad_three_field_bfgs=native_pad, one_step_unloaded_initialization=fast_init)
    m['object_primary_contact'] = object_primary
    m['line_search_disabled'] = no_line_search
    m['native_minimum_squared_residual'] = min_residual
    (out / 'input_audit.json').write_text(json.dumps(m, indent=2))

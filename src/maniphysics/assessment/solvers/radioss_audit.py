import json
import os
from pathlib import Path
import subprocess
import numpy as np
ROOT = Path(os.environ.get('MPB_RUNTIME_ROOT', 'runtime')).resolve()
from .radioss_result import fields, nodes
from .febio_deck import sections, FACES
from maniphysics.assessment.core import radioss as R

ANIMATION_CONVERTER = ROOT / 'anim_to_vtk_mass'

def expected_fraction(t, m):
    x = float(np.clip(t / m['duration_s'], 0, 1))
    grid = np.linspace(0, 1, 201)
    if m.get('continuous_velocity_control'):
        y = 30 * grid * grid * (1 - grid) ** 2
        y /= np.trapezoid(y, grid)
        i = min(int(x * 200), 199)
        value = float(np.trapezoid(y[:i + 1], grid[:i + 1]))
        value += (x - grid[i]) * (y[i] + np.interp(x, grid, y)) / 2
    else:
        y = grid ** 3 * (10 - 15 * grid + 6 * grid * grid)
        value = float(np.interp(x, grid, y))
    return value * m['closing_fraction']

def audit(out, frames=None):
    out = Path(out).resolve()
    m = json.loads((out / 'input_audit.json').read_text())
    c = m['source_case']
    padids = []
    crownids = []
    source_inp = Path(m['source']) / m.get('source_input_file', 'ccx.inp')
    if not source_inp.exists():
        candidates = [p for p in Path(m['source']).glob('*.inp') if p.name not in ('header.inp',) and (not p.name.startswith('stage_'))]
        if len(candidates) != 1:
            raise FileNotFoundError(f"expected one completed CCX input in {m['source']}, found {candidates}")
        source_inp = candidates[0]
    ss = sections(source_inp)
    for k in range(2):
        block = next((rows for (h, a, rows) in ss if h == '*ELEMENT' and a.get('ELSET') == f'EPAD{k}'))
        padids.append(sorted({int(v) for row in block for v in row.split(',')[1:]}))
        elements = {int(row.split(',')[0]): list(map(int, row.split(',')[1:])) for row in block}
        surface = next((rows for (h, a, rows) in ss if h == '*SURFACE' and a.get('NAME') == f'SPAD{k}'))
        crownids.append({elements[int(row.split(',')[0])][j] for row in surface for j in FACES[int(row.split(',')[1].strip()[1:])]})
    files = [out / x for x in frames] if frames else sorted(out.glob('modelA[0-9][0-9][0-9]'))
    rows = []
    for f in files:
        v = out / f'inertia_inspection.{os.getpid()}.tmp.vtk'
        with v.open('w') as log:
            subprocess.run([str(ANIMATION_CONVERTER), f.name], cwd=out, stdout=log, check=True)
        (t, a) = fields(v)
        v.unlink()
        ids = a['NODE_ID'].astype(int)
        assert len(np.unique(ids)) == len(ids)
        mass = a['NATIVE_NODAL_MASS']
        vel = a['Velocity']
        acc = a['Acceleration']
        cf = a['Contact_Forces']
        relative_v = vel.copy()
        pads = []
        fractions = []
        for (k, ni) in enumerate(padids):
            mask = np.isin(ids, ni)
            prescribed_v = nodes(a, 'Velocity', c['backs'][k]).mean(0)
            relative_v[mask] -= prescribed_v
            inertial = np.sum(mass[mask, None] * acc[mask], axis=0)
            contact = cf[mask].sum(0)
            outside = mask & ~np.isin(ids, list(crownids[k]))
            back = nodes(a, 'Reaction_Forces', c['backs'][k]).sum(0)
            scale = max(np.linalg.norm(back), np.linalg.norm(contact))
            residual = float(np.linalg.norm(back + contact - inertial))
            full = np.asarray(c['motion']['translations_m'][-1][k])
            disp = nodes(a, 'Displacement', c['backs'][k]).mean(0)
            fraction = float(disp @ full / (full @ full))
            fractions.append(fraction)
            pads.append(dict(back_N=back.tolist(), contact_N=contact.tolist(), inertial_N=inertial.tolist(), outside_source_contact_surface_force_N=cf[outside].sum(0).tolist(), outside_source_contact_surface_sum_force_magnitudes_N=float(np.linalg.norm(cf[outside], axis=1).sum()), back_displacement_m=disp.tolist(), observed_closing_fraction=fraction, prescribed_displacement_error_m=float(np.linalg.norm(disp - full * expected_fraction(t, m))), dynamic_balance_residual_N=residual, dynamic_balance_valid=bool(residual <= 0.001 + 0.05 * scale), inertial_force_small=bool(np.linalg.norm(inertial) <= 0.001 + 0.05 * scale)))
        obj = np.isin(ids, c['object_nodes'])
        obj_contact = cf[obj].sum(0)
        obj_inertial = np.sum(mass[obj, None] * acc[obj], axis=0)
        center_disp = nodes(a, 'Displacement', c['centering_nodes'])
        obj_support = -c['centering_N_m'] * center_disp.sum(0)
        obj_scale = max(np.linalg.norm(obj_contact), np.linalg.norm(obj_inertial), np.linalg.norm(obj_support))
        obj_residual = float(np.linalg.norm(obj_contact + obj_support - obj_inertial))
        pe = max((float(a[f'2DELEM_Plast_Strn_Layer_{j:3d}_'.replace(' ', '_')][(a['ELEMENT_ID'] >= 1) & (a['ELEMENT_ID'] <= m['shell_elements'])].max()) for j in range(1, m['nip'] + 1)))
        rows.append(dict(t=t, total_mass_kg=float(mass.sum()), observed_closing_fraction=float(np.mean(fractions)), expected_closing_fraction=expected_fraction(t, m), contact_force_N=float(min((abs(np.asarray(c['directions'][0]) @ np.asarray(p['contact_N'])) for p in pads))), nodal_kinetic_J=float(0.5 * np.sum(mass * np.sum(vel * vel, axis=1))), object_kinetic_J=float(0.5 * np.sum(mass[obj] * np.sum(vel[obj] * vel[obj], axis=1))), object_contact_N=obj_contact.tolist(), object_support_N=obj_support.tolist(), object_inertial_N=obj_inertial.tolist(), object_dynamic_balance_residual_N=obj_residual, object_dynamic_balance_valid=bool(obj_residual <= 0.001 + 0.05 * obj_scale), kinetic_excluding_prescribed_pad_translation_J=float(0.5 * np.sum(mass * np.sum(relative_v * relative_v, axis=1))), peeq=pe, pads=pads))
    r = dict(rows=rows)
    if not frames:
        (t, cols) = R.read_th(R.th_csv(out, 'model'))
        ie = np.asarray(cols['INTERNAL ENERGY'])
        kr = np.asarray(cols['ROTATION ENERGY'])
        kt = np.asarray(cols['KINETIC ENERGY'])
        for x in rows:
            x['internal_energy_J'] = float(np.interp(x['t'], t, ie))
            x['kinetic_excluding_prescribed_pad_translation_J'] += float(np.interp(x['t'], t, kr))
            x['relative_energy_small'] = bool(x['kinetic_excluding_prescribed_pad_translation_J'] <= 1e-08 + 0.05 * x['internal_energy_J'])
            x['native_translational_ke_J'] = float(np.interp(x['t'], t, kt))
        r['full_relative_energy_gate_passed'] = all((x['relative_energy_small'] for x in rows))
        r['full_pad_dynamic_balance_passed'] = all((p['dynamic_balance_valid'] for x in rows for p in x['pads']))
        r['full_pad_inertial_force_gate_passed'] = all((p['inertial_force_small'] for x in rows for p in x['pads']))
        r['full_object_dynamic_balance_passed'] = all((x['object_dynamic_balance_valid'] for x in rows))
        limit = c['limit']
        crossing = next((x for x in rows if x['peeq'] >= limit), None)
        onset_window = [] if crossing is None else [x for x in rows if x['peeq'] > 0 and x['t'] <= crossing['t']]

        def ratio(value, scale):
            return float(value / max(scale, 1e-12))
        r['onset_window'] = dict(samples=len(onset_window), crossing_time_s=None if crossing is None else crossing['t'], relative_energy_passed=bool(onset_window and all((x['relative_energy_small'] for x in onset_window))), pad_dynamic_balance_passed=bool(onset_window and all((p['dynamic_balance_valid'] for x in onset_window for p in x['pads']))), pad_inertial_force_passed=bool(onset_window and all((p['inertial_force_small'] for x in onset_window for p in x['pads']))), object_dynamic_balance_passed=bool(onset_window and all((x['object_dynamic_balance_valid'] for x in onset_window))), interface_force_surface_exclusive_passed=bool(onset_window and all((p['outside_source_contact_surface_sum_force_magnitudes_N'] <= 1e-06 + 0.001 * x['contact_force_N'] for x in onset_window for p in x['pads']))), maximum_relative_ke_over_ie=None if not onset_window else max((ratio(x['kinetic_excluding_prescribed_pad_translation_J'], x['internal_energy_J']) for x in onset_window)), maximum_pad_inertia_over_contact_force=None if not onset_window else max((ratio(np.linalg.norm(p['inertial_N']), x['contact_force_N']) for x in onset_window for p in x['pads'])), maximum_object_balance_residual_N=None if not onset_window else max((x['object_dynamic_balance_residual_N'] for x in onset_window)), maximum_outside_source_contact_surface_force_N=None if not onset_window else max((p['outside_source_contact_surface_sum_force_magnitudes_N'] for x in onset_window for p in x['pads'])))
        r['onset_window']['all_gates_passed'] = bool(all((r['onset_window'][k] for k in ('relative_energy_passed', 'pad_dynamic_balance_passed', 'pad_inertial_force_passed', 'object_dynamic_balance_passed'))))
        r['onset_window']['interface_force_gates_passed'] = bool(all((r['onset_window'][k] for k in ('relative_energy_passed', 'pad_dynamic_balance_passed', 'object_dynamic_balance_passed', 'interface_force_surface_exclusive_passed'))))
        if crossing is not None:
            i = rows.index(crossing)
            lo = rows[max(0, i - 1)]
            (a, b) = (lo['peeq'] / limit, crossing['peeq'] / limit)
            estimate = crossing['contact_force_N'] if b == a else lo['contact_force_N'] + (1 - a) * (crossing['contact_force_N'] - lo['contact_force_N']) / (b - a)
            r['interface_force_crossing'] = dict(force_interval_N=[lo['contact_force_N'], crossing['contact_force_N']], estimate_N=float(estimate), time_interval_s=[lo['t'], crossing['t']])
        (out / 'inertia_audit.json').write_text(json.dumps(r, indent=2))
    return r

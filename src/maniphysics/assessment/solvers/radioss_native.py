import os
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get('MPB_RUNTIME_ROOT', 'runtime')).resolve()
from . import radioss_deck as base
from . import radioss_result as direct
from maniphysics.assessment.core import radioss as R
OUT = ROOT

def smooth(x):
    x = np.clip(x, 0, 1)
    return x ** 3 * (10 - 15 * x + 6 * x * x)

def prepare(case, tag, duration, fraction, hold, dt_min, threads, source=None, dyrel=0, velocity=False):
    source = Path(source)
    out = OUT / case / tag
    m = base.translate(source, out, duration=duration, fraction=fraction, dt_min=dt_min)
    p = out / 'model_0000.rad'
    lines = p.read_text().splitlines()
    i = lines.index('/FUNCT/1')
    j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith('/')))
    x = np.linspace(0, 1, 201)
    values = 30 * x * x * (1 - x) ** 2 if velocity else smooth(x)
    integral = float(np.trapezoid(values, x)) if velocity else None
    if velocity:
        values /= integral
    curve = [(float(a), float(b)) for (a, b) in zip(x, values)]
    curve.append((1000, 0 if velocity else 1))
    lines[i:j] = R.c_funct(1, curve, 'continuous velocity' if velocity else 'quintic smooth closing ramp then hold')
    if velocity:
        for k in range(len(lines)):
            if lines[k].startswith('/IMPDISP/'):
                lines[k] = lines[k].replace('/IMPDISP/', '/IMPVEL/')
                if int(lines[k + 2][:10]) == 1:
                    row = lines[k + 3]
                    lines[k + 3] = row[:20] + R.f20(float(row[20:40]) / duration) + row[40:]
    p.write_text('\n'.join(lines) + '\n')
    end = duration * (1 + hold)
    engine = R.engine('model', end, dt_th=duration / 1000, dt_anim=duration / 120, anim=['SHELL/EPSP/ALL', 'VECT/DISP', 'VECT/FREAC', 'VECT/CONT', 'VECT/VEL', 'VECT/ACC', 'MASS', 'SHELL/TENS/STRESS/MEMB'], dt_min=dt_min)
    if dyrel:
        engine += '/DYREL\n' + R.f20(1) + R.f20(dyrel) + '\n'
    (out / 'model_0001.rad').write_text(engine)
    m.update(hold_fraction=hold, requested_end_s=end, threads=threads, dynamic_relaxation_period_s=dyrel, continuous_velocity_control=velocity, velocity_table_integral_before_normalization=integral, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    m['executed_inputs_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.rad')}
    (out / 'input_audit.json').write_text(json.dumps(m, indent=2))
    return out

def analyze(out):
    r = direct.analyze(out)
    m = json.loads((out / 'input_audit.json').read_text())
    (t, cols) = R.read_th(R.th_csv(out, 'model'))
    ie = np.asarray(cols['INTERNAL ENERGY'])
    ke = np.asarray(cols['KINETIC ENERGY']) + np.asarray(cols['ROTATION ENERGY'])
    energy_ok = np.isfinite(ie) & np.isfinite(ke) & (ie >= 0) & (ke <= 1e-08 + 0.05 * ie)
    np.savez(out / 'energy_history.npz', t=t, ie=ie, ke=ke, energy_ok=energy_ok)
    lc = r['execution']['last_cycle']
    added = 100 * lc['added'] / (lc['mass'] - lc['added'])
    material_limit = m['source_case']['source']['material_domain']['max_peeq']
    for row in r['rows']:
        row['closing_fraction_of_ccx_full_ramp'] = float(smooth(row['t'] / m['duration_s']) * m['closing_fraction'])
        j = min(int(np.searchsorted(t, row['t'])), len(t) - 1)
        row['reaction_quality_valid'] = row['valid']
        row['material_domain_valid'] = bool(row['peeq'] <= material_limit)
        row['energy_prefix_valid'] = bool(energy_ok[:j + 1].all())
        row['ke_J'] = float(np.interp(row['t'], t, ke))
        row['ie_J'] = float(np.interp(row['t'], t, ie))
        row['valid'] = bool(row['valid'] and row['energy_prefix_valid'] and (added <= 2) and row['material_domain_valid'])
    raw_hit = next((i for (i, x) in enumerate(r['rows']) if x['indicator_ratio'] >= 1), None)
    r['raw_indicator_crossing'] = None if raw_hit is None else r['rows'][raw_hit]
    prefix = []
    for row in r['rows']:
        if not row['valid']:
            break
        prefix.append(row)
    hit = next((i for (i, row) in enumerate(prefix) if row['indicator_ratio'] >= 1), None)
    r.update(status='unknown', threshold=None, valid_prefix_rows=len(prefix), added_mass_percent_of_original=added, first_energy_invalid_s=None if energy_ok.all() else float(t[np.where(~energy_ok)[0][0]]))
    if hit is not None:
        a = prefix[hit]
        b = prefix[max(0, hit - 1)]
        r.update(status='damaged_in_validated_prefix', threshold=dict(force_interval_N=[b['F'], a['F']], closing_fraction_interval=[b['closing_fraction_of_ccx_full_ramp'], a['closing_fraction_of_ccx_full_ramp']]))
    elif r['execution']['completed'] and len(prefix) == len(r['rows']) and prefix:
        r['status'] = 'safe_in_calculated_range'
    (out / 'result.json').write_text(json.dumps(r, indent=2))
    return r

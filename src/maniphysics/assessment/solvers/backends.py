import json
import os
import signal
import subprocess
import time
from types import SimpleNamespace
import numpy as np
from ..cache import save_json


def row(t, force, q, indicator, valid, reason=None):
    return dict(time_s=float(t), force_N=float(force), closure_m=max(0., float(q)),
                indicator=float(indicator), valid=bool(valid), reason=reason)


def calculix(case_dir, output, profile, runtime):
    from ..core import contact_history, reference_contact
    runtime.bind('calculix')
    code, execution = reference_contact.run_virgin_prefix(case_dir, threads=runtime.threads,
                                                         timeout=runtime.timeout_s)
    result = contact_history.analyze(case_dir, code)
    case = json.loads((case_dir / 'case.json').read_text())
    directions = np.asarray(case['directions'])
    rows = []
    initial = result.get('reference_initialization') or {}
    if initial.get('status') == 'invalid':
        return dict(rows=[], execution=execution, reason=initial.get('reason'), completed=False)
    t0 = case['source']['reference_initialization']['end_time_s']
    gap = 2 * profile['spec'].get('contact_initial_gap_m', 2e-6)
    for r in result['rows']:
        if r['t'] <= t0:
            continue
        if 'jaw_mean_displacement_m' not in r or r.get('indicator_ratio') is None:
            rows.append(row(r['t'], 0, 0, 0, False, r.get('reason', 'incomplete_fields')))
            break
        q = np.sum(np.asarray(r['jaw_mean_displacement_m']) * directions) - gap
        rows.append(row(r['t'], r['F'], q, r['indicator_ratio'], r['valid'], r.get('reason')))
        if not r['valid'] or r['indicator_ratio'] >= 1:
            break
    return dict(rows=rows, execution=execution, completed=result['requested_path_completed'])


def febio_options(mid):
    options = dict(mode='baseline', fraction=.5, dt=.02, solver=None, penalty=.03,
        maxaug=100, gaptol=1e-5, contact_offset=0, node_reloc=False, two_pass=False,
        seg_up=None, tether=100, search_radius=.001, friction_penalty=2e11,
        lsmin=None, initial_step=.002, aggressive=True, max_retries=12, opt_iter=100,
        surface_rule=None, pad_three_field=False, length_scale=1, dynamic=True,
        ramp_duration=1., density=1000., object_density=1000., pad_density=1000.,
        rtol=1e-10, dtol=0., etol=.001)
    if mid == 'strawberry':
        options.update(dt=.005, ramp_duration=5., gaptol=5e-6, rtol=1e-8, dtol=.001)
    if mid == 'plum':
        options.update(dynamic=True, fraction=1., dt=.01, ramp_duration=5., gaptol=5e-6, rtol=1e-8, penalty=.3)
    return options


def febio(case_dir, output, profile, runtime):
    from . import febio_native as native, febio_run as fd
    from .febio_log import read_log
    from .febio_prefix import valid_prefix
    files = runtime.bind('febio')
    mid = json.loads((case_dir / 'case.json').read_text())['source']['model_id']
    native.OUT = output
    native.contact('native', 'staged', source=case_dir,
                   penalty=10., auto_penalty=True, gaptol=1e-6, native_pad=True,
                   min_residual=1e-8 if mid == 'plum' else None)
    fd.OUT = output
    options = febio_options(mid)
    args = SimpleNamespace(**options, tag='febio', source=output / 'native/staged',
                           timeout=runtime.timeout_s)
    run_dir = fd.prepare(args)
    env = runtime.environment()
    command = [str(files['solver']), *runtime.febio_arguments, '-import', str(files['plugin']), '-i', 'model.feb']
    start = time.monotonic()
    with (run_dir / 'stdout.log').open('w') as log:
        p = subprocess.Popen(command, cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            rc = p.wait(timeout=runtime.timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            rc = 124
    execution = dict(returncode=rc, elapsed_s=time.monotonic() - start)
    save_json(run_dir / 'execution.json', execution)
    result = fd.analyze(run_dir, execution)
    valid, reason = valid_prefix(result)
    case = json.loads((case_dir / 'case.json').read_text())
    directions = np.asarray(case['directions'])
    backs = [dict(read_log(run_dir / f'back{k}.txt')) for k in (0, 1)] if valid else []
    rows = []
    for r in valid:
        u = np.asarray([b[r['t']][:, 1:4].mean(0) for b in backs])
        q = np.sum(u * directions) - 2 * profile['spec'].get('contact_initial_gap_m', 2e-6)
        rows.append(row(r['t'], r['F'], q, r['indicator_ratio'], True))
        if r['indicator_ratio'] >= 1:
            break
    return dict(rows=rows, completed=result['completed'], execution=execution, prefix_reason=reason)


def openradioss(case_dir, output, profile, runtime):
    from . import radioss_native as native, radioss_deck, radioss_audit
    from .radioss_evidence import bilateral, _frame_gates, _run_gates
    runtime.bind('openradioss')
    case = json.loads((case_dir / 'case.json').read_text())
    cfg = profile['settings']
    full = np.asarray(case['motion']['translations_m'][-1])
    total = float(np.sum(full * np.asarray(case['directions'])))
    points = np.asarray(case['source']['initial_contacts_m'])
    wall = (np.linalg.norm(points[1] - points[0]) - 2 * case['thickness_m']) / total
    functional = case['source']['model_id'] == 'portion-cup-1oz-pp'
    fraction = min(2., round(wall * 1.1, 3)) if cfg['closing_fraction'] == 'wall' else cfg['closing_fraction']
    duration = cfg['duration_s'] * fraction if cfg['closing_fraction'] == 'wall' else cfg['duration_s']
    native.OUT = output
    tmp = output / 'tmp'
    tmp.mkdir()
    os.environ['TMPDIR'] = str(tmp)
    os.environ['RAD_TMPDIR'] = str(tmp)
    runs = []
    for multiplier in (1., 2., 4.):
        run_dir = native.prepare('radioss', f'ramp{multiplier:g}', duration * multiplier,
            fraction, cfg['hold_fraction'], 0., runtime.threads, source=case_dir,
            velocity=cfg['continuous_velocity_control'])
        execution = radioss_deck.run(run_dir, timeout=runtime.timeout_s, threads=runtime.threads)
        if not (run_dir / 'modelT01').exists() or not list(run_dir.glob('modelA[0-9][0-9][0-9]')):
            runs.append(dict(rows=[], execution=execution, completed=False, reason='solver_output_missing'))
            break
        result = native.analyze(run_dir)
        audit = radioss_audit.audit(run_dir)
        limit = profile['spec']['material_domain']['max_peeq']
        bad = _run_gates(audit, result, limit)
        rows = []
        started = False
        stop = None
        for r in audit['rows']:
            if not started:
                if not bilateral(r):
                    if r['peeq'] >= case['limit']:
                        stop = 'damage_before_bilateral_contact'
                        break
                    continue
                started = True
            why = _frame_gates(r, energy=r['peeq'] > 0)
            if bad:
                why = ','.join(bad)
            if r['peeq'] > limit:
                why = 'material_domain_exceeded'
            qraw = np.sum(np.asarray([p['back_displacement_m'] for p in r['pads']]) * np.asarray(case['directions']))
            ratio = r['peeq'] / case['limit']
            if functional:
                ratio = max(ratio, r['observed_closing_fraction'] / wall)
            rows.append(row(r['t'], r['contact_force_N'], qraw - 2 * profile['spec'].get('contact_initial_gap_m', 2e-6), ratio, why is None, why))
            if why:
                stop = why
                break
            if ratio >= 1:
                break
        record = dict(rows=rows, completed=execution.get('completed', False), execution=execution,
                      reason=stop, ramp_multiplier=multiplier, functional_wall_contact=functional)
        runs.append(record)
        if stop not in ('relative_kinetic_energy', 'pad_dynamic_balance', 'object_dynamic_balance'):
            break
    selected = max(runs, key=lambda x: (any(r['valid'] and r['indicator'] >= 1 for r in x['rows']),
                                       max((r['force_N'] for r in x['rows'] if r['valid']), default=0.)))
    selected['attempts'] = [dict(ramp_multiplier=x.get('ramp_multiplier'), reason=x.get('reason'),
                                 valid_rows=sum(r['valid'] for r in x['rows'])) for x in runs]
    return selected

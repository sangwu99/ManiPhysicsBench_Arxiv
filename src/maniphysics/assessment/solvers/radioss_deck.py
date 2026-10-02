import os
import hashlib, json, subprocess, time
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get('MPB_RUNTIME_ROOT', 'runtime')).resolve()
from maniphysics.assessment.core import radioss as R
from .febio_deck import sections
(I, F) = (R.i10, R.f20)
RHO = {'yakult-bottle': 1040.0, 'paper-cup-takeaway': 750.0, 'paper-cup-2oz': 750.0, 'portion-cup-1oz-pp': 905.0}

def clamped_table(table):
    table = [list(p) for p in table]
    table.append([table[-1][0], max(1.0, 2 * table[-1][1])])
    return table
SUBTRI = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])

def ogden(mid, E, nu, rho):
    G = E / (2 * (1 + nu))
    return [f'/MAT/LAW42/{mid}', 'neo Hookean pad', F(rho), F(nu) + F(0) + I(0) + I(0) + F(1) + I(0) + I(1), F(G) + F(0) * 4, '', F(2) + F(0) * 4, '']

def solidprop(pid):
    return [f'/PROP/TYPE14/{pid}', 'tetra10 pad', I(24) + I(0) + ' ' * 10 + I(0) + ' ' * 10 + I(0) + I(1000) + I(0) + F(0), F(0) * 5, F(0) + I(0) + I(0), I(0) + I(0)]

def springprop(pid, k):
    a = [f'/PROP/TYPE8/{pid}', 'Cartesian ground spring', F(1e-08) + F(1e-16) + I(0) * 6]
    for d in range(6):
        a += [F(k if d < 3 else 0) + F(0) * 4, I(0) * 5 + ' ' * 10 + F(0) * 2, F(0) * 4]
    return a + [I(0) + F(0)]

def tetra10(pid, elements):
    a = [f'/TETRA10/{pid}']
    for (eid, n) in elements:
        a += [I(eid), ''.join((I(v) for v in n))]
    return a

def contact(cid, secondary, main, mu, stfac):
    return [f'/INTER/TYPE24/{cid}', 'deformable pad to shell', I(secondary) + I(main) + I(5) + ' ' * 20 + I(3) + ' ' * 10 + I(1000) + ' ' * 10 + I(0), I(0) + ' ' * 20 + I(1000) + F(0) + F(0) + F(0), F(0) + F(0) + I(1000) + I(1000) + F(0) + F(0), F(stfac) + F(mu) + ' ' * 20 + F(0) + F(0), '       000' + ' ' * 20 + I(-1) + F(0) + ' ' * 20 + F(0), I(0) + I(0) + F(0) + ' ' * 10 + I(0) + F(0) + ' ' * 10 + I(0)]

def translate(source, out, *, duration=0.05, fraction=0.1, stfac=1.0, nip=5, dt_min=0.0):
    out.mkdir(parents=True, exist_ok=False)
    case = json.loads((source / 'case.json').read_text())
    assert case['shell'] and case['criterion'] == 'peeq'
    source_inp = source / 'ccx.inp'
    if not source_inp.exists():
        candidates = [p for p in source.glob('*.inp') if p.name not in ('header.inp',) and (not p.name.startswith('stage_'))]
        if len(candidates) != 1:
            raise FileNotFoundError(f'expected one completed CCX input in {source}, found {candidates}')
        source_inp = candidates[0]
    ss = sections(source_inp)
    nodes = {}
    els = {}
    groups = {}
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
    nsource = len(nodes)
    grounds = []
    for n in case['centering_nodes']:
        g = max(nodes) + 1
        nodes[g] = nodes[n].copy()
        grounds.append(g)
    assert sorted(nodes) == list(range(1, len(nodes) + 1))
    mid = case['source']['model_id']
    mat = dict(case['material'])
    mat['rho'] = mat.get('rho', RHO[mid])
    mat['plastic'] = clamped_table(mat['plastic'])
    p = case['pad']
    L = R.c_begin('model') + R.c_mat(1, mat, 'object shell') + ogden(2, p['E_Pa'], p['nu'], 1000.0) + R.c_nodes(np.array(list(nodes.values())))
    L += R.c_part(1, 1, 1, 'object wall') + R.c_prop_shell(1, case['thickness_m'], ish3n=30, nip=nip)
    tris = []
    parent = []
    for eid in groups['EOBJ']:
        for t in np.array(els[eid])[SUBTRI]:
            tris.append(t)
            parent.append(eid)
    L += R.c_sh3n(1, tris) + solidprop(2)
    for k in (0, 1):
        pid = k + 2
        L += R.c_part(pid, 2, 2, f'pad{k}') + tetra10(pid, [(i + len(tris), els[i]) for i in groups[f'EPAD{k}']])
    L += springprop(4, case['centering_N_m']) + R.c_part(4, 4, 0, 'ground springs') + ['/SPRING/4']
    for (i, (n, g)) in enumerate(zip(case['centering_nodes'], grounds), 1):
        L.append(I(10000000 + i) + I(n) + I(g) + I(0) * 4 + ' ' * 20 + I(0))
    L += R.c_grnod(90, grounds, 'ground') + ['/BCS/90', 'fixed ground', '   111 111' + I(0) + I(90)]
    L += R.c_funct(1, [(0, 0), (1, 1), (1000, 1)], 'linear closing ramp') + R.c_funct(2, [(0, 0), (1000, 0)], 'zero')
    for k in (0, 1):
        gid = k + 1
        L += R.c_grnod(gid, case['backs'][k], f'back{k}')
        for (d, u) in enumerate(case['motion']['translations_m'][-1][k]):
            L += R.c_impdisp(k * 3 + d + 1, 1 if u else 2, 'XYZ'[d], gid, duration, u * fraction if u else 1.0, f'back{k}d{d}')
        L += R.c_th('NODE', 91 + k, case['backs'][k], R.TH_NODE_VARS, f'back{k}')
    L += R.c_th('NODE', 93, case['centering_nodes'], R.TH_NODE_VARS, 'center') + R.c_th('PART', 94, [1, 2, 3], ['DEF'], 'parts')
    for pid in (1, 2, 3):
        surface = R.c_surf_part(pid, [pid], f'part{pid}surface')
        if pid > 1:
            surface[0] = f'/SURF/PART/EXT/{pid}'
        L += surface
    for k in (0, 1):
        L += contact(k + 1, k + 2, 1, case['friction'], stfac)
    L += ['/END']
    starter = '\n'.join(L) + '\n'
    engine = R.engine('model', duration, dt_th=duration / 1000, dt_anim=duration / 80, anim=['SHELL/EPSP/ALL', 'VECT/DISP', 'SHELL/TENS/STRESS/MEMB'], dt_min=dt_min)
    R.write_case(out, 'model', starter, engine)
    meta = dict(source=str(source), source_input_file=source_inp.name, source_case=case, source_input_sha256=hashlib.sha256(source_inp.read_bytes()).hexdigest(), duration_s=duration, closing_fraction=fraction, contact_stfac=stfac, nip=nip, rho_object=mat['rho'], rho_pad=1000.0, shell_elements=len(tris), source_shell_elements=len(groups['EOBJ']), source_nodes=nsource, source_shell_parent=parent, nodal_minimum_timestep_s=dt_min, mass_scaling_enabled=dt_min > 0, mass_added_acceptance_percent=2.0, regularization_spring_mass_kg=1e-08, regularization_spring_inertia_kg_m2=1e-16)
    (out / 'input_audit.json').write_text(json.dumps(meta, indent=2))
    return meta

def run(out, timeout=1800, *, threads=2):
    import signal
    env = R.env()
    env.update(OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1')
    exe = Path(R.EXEC) / 'engine_linux64_gf'
    start = time.monotonic()
    rc = 0
    stages = {}
    for (label, binary, inp) in [('starter', Path(R.EXEC) / 'starter_linux64_gf', 'model_0000.rad'), ('engine', exe, 'model_0001.rad')]:
        cmd = ['time', '-v', '-o', f'{label}_resources.txt', str(binary), '-i', inp, '-nt', str(threads)]
        if label == 'starter':
            cmd += ['-np', '1']
        with (out / (label + '_console.log')).open('w') as log:
            proc = subprocess.Popen(cmd, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                rc = 124
        native = out / ('model_0000.out' if label == 'starter' else 'model_0001.out')
        text = (native.read_text() if native.exists() else '') + '\n' + (out / (label + '_console.log')).read_text()
        (out / (label + '.log')).write_text(text)
        stages[label] = dict(returncode=rc)
        if rc or (label == 'starter' and R.starter_errors(text)):
            break
    s = (out / 'starter.log').read_text()
    e = (out / 'engine.log').read_text() if (out / 'engine.log').exists() else ''
    summary = dict(returncode=rc, elapsed_s=time.monotonic() - start, threads=threads, stages=stages, starter_errors=R.starter_errors(s), last_cycle=R.last_cycle(e), added_mass_percent=R.mass_added(e))
    summary['completed'] = bool(rc == 0 and (not any((v in e for v in ['**ERROR', '** ERROR', 'Fatal error', 'ERROR TERMINATION']))) and R.last_cycle(e) and np.isfinite(R.last_cycle(e)['ke']))
    if summary['completed']:
        (t, _) = R.read_th(R.th_csv(out, 'model'))
        summary['last_observed_time_s'] = float(t[-1])
        engine = (out / 'model_0001.rad').read_text().splitlines()
        stop = float(engine[engine.index('/RUN/model/1') + 1])
        dt_th = float(engine[engine.index('/TFILE/4') + 1])
        summary['last_output_gap_s'] = float(stop - t[-1])
        summary['completed'] = bool(t[-1] >= stop - dt_th * 1.01)
    (out / 'execution.json').write_text(json.dumps(summary, indent=2))
    return summary

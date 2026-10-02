import glob
import os
import re
import subprocess
from pathlib import Path
import numpy as np
ROOT = os.environ.get('OPENRADIOSS_PATH', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tools', 'openradioss', 'OpenRadioss'))
EXEC = os.path.join(ROOT, 'exec')
INVERS = 2025
BIG = 1e+30
(RAMP_FCT, ZERO_FCT) = (1, 2)
TH_NODE_VARS = ['DX', 'DY', 'DZ', 'REACX', 'REACY', 'REACZ']
TH_MONV_VARS = ['VOL', 'P', 'A']

def env():
    lib = ':'.join([os.path.join(ROOT, 'extlib', 'hm_reader', 'linux64'), os.path.join(ROOT, 'extlib', 'h3d', 'lib', 'linux64'), os.environ.get('LD_LIBRARY_PATH', '')])
    return dict(os.environ, OPENRADIOSS_PATH=ROOT, RAD_CFG_PATH=os.path.join(ROOT, 'hm_cfg_files'), RAD_H3D_PATH=os.path.join(ROOT, 'extlib', 'h3d', 'lib', 'linux64'), LD_LIBRARY_PATH=lib, OMP_STACKSIZE='400m')

def i10(v):
    return f'{int(v):10d}'

def f20(v):
    return f'{float(v):20.10g}'

def _rows(vals, per=10, w=i10):
    v = list(vals)
    return [''.join((w(x) for x in v[k:k + per])) for k in range(0, len(v), per)]

def c_begin(runname, invers=INVERS):
    u = f"{'kg':>20s}{'m':>20s}{'s':>20s}"
    return ['#RADIOSS STARTER', '/BEGIN', f'{runname:<100s}', i10(invers), u, u]

def c_nodes(nodes):
    return ['/NODE'] + [i10(k + 1) + f20(p[0]) + f20(p[1]) + f20(p[2]) for (k, p) in enumerate(nodes)]

def c_sh3n(part_id, tris):
    return [f'/SH3N/{part_id}'] + [i10(k + 1) + i10(t[0]) + i10(t[1]) + i10(t[2]) for (k, t) in enumerate(tris)]

def c_part(pid, prop_id, mat_id, title='part'):
    return [f'/PART/{pid}', f'{title:<100s}', i10(prop_id) + i10(mat_id) + i10(0)]

def c_prop_shell(pid, thick, *, ish3n=30, nip=5, title='shell'):
    return [f'/PROP/TYPE1/{pid}', f'{title:<100s}', i10(0) + i10(0) + i10(ish3n) + i10(0) + i10(0) + ' ' * 10 + f20(0), f20(0) * 5, i10(nip) + ' ' * 10 + f20(thick) + f20(0) + ' ' * 10 + i10(0) + i10(0) + i10(0)]

def c_mat(mid, m, title='mat'):
    if 'plastic' in m:
        table = np.asarray(m['plastic'], float)
        if table.ndim != 2 or table.shape[1] != 2 or len(table) < 2 or (not np.isfinite(table).all()) or (table[0, 1] != 0) or np.any(np.diff(table[:, 1]) <= 0) or np.any(table[:, 0] <= 0):
            raise ValueError('plastic must be ordered (positive stress, plastic strain) pairs starting at zero strain')
        fid = 1000 + mid
        return [f'/MAT/LAW36/{mid}', f'{title:<100s}', f20(m['rho']), f20(m['E']) + f20(m['nu']) + f20(0) * 3, i10(1) + i10(0) + f20(0) * 3 + ' ' * 10 + i10(0), i10(0) + f20(0) + i10(0) + f20(0) * 2, i10(fid), f20(1), f20(0)] + c_funct(fid, [(ep, sig) for (sig, ep) in table], 'true stress versus equivalent plastic strain')
    if m.get('sig_y') is None:
        return [f'/MAT/LAW1/{mid}', f'{title:<100s}', f20(m['rho']), f20(m['E']) + f20(m['nu'])]
    return [f'/MAT/LAW2/{mid}', f'{title:<100s}', f20(m['rho']), f20(m['E']) + f20(m['nu']) + i10(0) + i10(0) + f20(0), f20(m['sig_y']) + f20(m.get('b', 0.0)) + f20(m.get('n', 1.0)) + f20(m.get('eps_max', 0.0)) + f20(m.get('sig_max', 0.0)), f20(0) + f20(0) + i10(0) + i10(0) + f20(0) + f20(0), f20(0) * 5]

def c_grnod(gid, ids, title='grn'):
    return [f'/GRNOD/NODE/{gid}', f'{title:<100s}'] + _rows(ids)

def c_surf_part(sid, part_ids, title='closed surface'):
    return [f'/SURF/PART/{sid}', f'{title:<100s}'] + _rows(part_ids)

def c_funct(fid, xy, title='f'):
    return [f'/FUNCT/{fid}', f'{title:<100s}'] + [f20(x) + f20(y) for (x, y) in xy]

def c_impdisp(lid, fid, axis, gnod, scale_x, scale_y, title='impdisp'):
    return [f'/IMPDISP/{lid}', f'{title:<100s}', i10(fid) + f'{axis:>10s}' + i10(0) + i10(0) + i10(gnod), f20(scale_x) + f20(scale_y) + f20(0) + f20(BIG)]

def c_monvol_lfluid(vid, surf_id, fl, title='sealed liquid'):
    return [f'/MONVOL/LFLUID/{vid}', f'{title:<100s}', i10(surf_id), f20(0) + f20(0), f20(fl['rho']), i10(0) + i10(0) + f20(fl['K']) + f20(0), i10(0) + i10(0) + f20(0) + f20(0), i10(0) + i10(0) + f20(fl.get('p_add', 0.0)) + f20(fl.get('p_max', 0.0))]

def c_monvol_gas(vid, surf_id, g, title='sealed gas'):
    return [f'/MONVOL/GAS/{vid}', f'{title:<100s}', i10(surf_id) + i10(0), f20(0) * 5, f20(g['gamma']) + f20(g.get('mu', 0.0)) + f20(g.get('t_relax', 0.0)) + f20(g.get('t_ini', 0.0)) + f20(g.get('rho', 0.0)), f20(g['p_ext']) + f20(g['p_ini']) + f20(g.get('p_max', 0.0)) + f20(g.get('v_inc', 0.0)) + f20(g.get('m_ini', 0.0)), i10(0)]

def c_th(kind, tid, ids, vars_, title='th'):
    head = [f'/TH/{kind}/{tid}', f'{title:<100s}', ''.join((f'{v:<10s}' for v in vars_))]
    if kind in ('MONVOL', 'PART'):
        return head + _rows(ids)
    return head + [i10(n) + i10(0) + ' ' * 80 for n in ids]

def starter(runname, nodes, tris, groups, *, mat, thick, imp, t_ramp, fluid=None, gas=None, th_nodes=None, ish3n=30, nip=5, invers=INVERS, title=''):
    L = c_begin(runname, invers) + ['/TITLE', f'{title or runname:<100s}']
    L += c_mat(1, mat, 'shell material')
    L += c_nodes(nodes)
    L += c_part(1, 1, 1, 'container wall')
    L += c_sh3n(1, tris)
    L += c_prop_shell(1, thick, ish3n=ish3n, nip=nip)
    gid = {}
    for (k, (name, ids)) in enumerate(groups.items(), start=1):
        gid[name] = k
        L += c_grnod(k, ids, name)
    L += c_funct(RAMP_FCT, [(0.0, 0.0), (1.0, 1.0), (1000.0, 1.0)], 'ramp 0->1')
    L += c_funct(ZERO_FCT, [(0.0, 0.0), (1000.0, 0.0)], 'constant 0')
    lid = 0
    for (name, vec) in imp:
        for (ax, v) in zip('XYZ', vec):
            if v is None:
                continue
            lid += 1
            (f, sy) = (RAMP_FCT, v) if v else (ZERO_FCT, 1.0)
            L += c_impdisp(lid, f, ax, gid[name], t_ramp, sy, f'{name}_{ax}')
    if fluid is not None or gas is not None:
        L += c_surf_part(1, [1], 'container closed surface')
        if fluid is not None:
            L += c_monvol_lfluid(1, 1, fluid)
        else:
            L += c_monvol_gas(1, 1, gas)
        L += c_th('MONVOL', 90, [1], TH_MONV_VARS, 'monvol')
    watch = th_nodes if th_nodes is not None else np.unique(np.concatenate([np.asarray(v) for v in groups.values()]))
    L += c_th('NODE', 91, watch, TH_NODE_VARS, 'patch')
    L += c_th('PART', 92, [1], ['DEF'], 'wall')
    L += ['/END']
    return '\n'.join(L) + '\n'

def engine(runname, t_stop, *, dt_th=None, dt_anim=None, dt_min=0.0, tsca=0.9, invers=INVERS, anim=None):
    L = ['/VERS/' + str(invers), f'/RUN/{runname}/1', f20(t_stop), '/TFILE/4', f20(dt_th if dt_th else t_stop / 500.0)]
    if dt_anim:
        L += ['/ANIM/DT', f20(0.0) + f20(dt_anim)]
        if anim is None:
            L += ['/ANIM/GZIP', '/ANIM/SHELL/TENS/STRESS/MEMB', '/ANIM/VECT/DISP']
        else:
            L += [f'/ANIM/{a}' for a in anim]
    L += ['/DT', f20(tsca) + f20(0.0)]
    if dt_min > 0:
        L += ['/DT/NODA/CST', f20(tsca) + f20(dt_min)]
    L += ['/MON/ON', '/PRINT/-1000']
    return '\n'.join(L) + '\n'

def write_case(work, runname, starter_txt, engine_txt):
    os.makedirs(work, exist_ok=True)
    if Path(work, f'{runname}_0000.rad').exists():
        archive_case(work, runname, include_inputs=True)
    s = os.path.join(work, f'{runname}_0000.rad')
    e = os.path.join(work, f'{runname}_0001.rad')
    open(s, 'w').write(starter_txt)
    open(e, 'w').write(engine_txt)
    return (s, e)
STALE = ('_0001.rst', '_0000_0001.rst', 'T01', 'T01.csv', 'A001', 'A002')

def archive_case(work, runname, *, include_inputs=False):
    import datetime
    import shutil
    import uuid
    root = Path(work)
    patterns = [runname + '_000*', runname + 'A[0-9][0-9][0-9]*', runname + 'T[0-9][0-9]*', runname + '_TITLES', runname + '.*']
    paths = sorted({p for pattern in patterns for p in root.glob(pattern) if p.is_file()})
    if not include_inputs:
        paths = [p for p in paths if p.suffix != '.rad' and p.name != runname + '.meta.npz']
    if not paths:
        return None
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
    dest = root / 'archive' / runname / (stamp + '_' + uuid.uuid4().hex[:8])
    dest.mkdir(parents=True)
    for p in paths:
        shutil.move(str(p), dest / p.name)
    return str(dest)

def _clean(work, runname):
    archive_case(work, runname)

def run(work, runname, *, nt=2, timeout=14400):
    _clean(work, runname)
    E = env()
    E['OMP_NUM_THREADS'] = str(nt)
    try:
        r0 = subprocess.run([os.path.join(EXEC, 'starter_linux64_gf'), '-i', f'{runname}_0000.rad', '-np', '1', '-nt', str(nt)], cwd=work, capture_output=True, text=True, timeout=timeout, env=E)
    except subprocess.TimeoutExpired:
        p = Path(work, f'{runname}_0000.out')
        return (124, (p.read_text() if p.exists() else '') + '\nSTARTER TIMEOUT', '')
    out0 = os.path.join(work, f'{runname}_0000.out')
    log0 = open(out0).read() if os.path.exists(out0) else r0.stdout + r0.stderr
    if r0.returncode or starter_errors(log0):
        return (1, log0, '')
    try:
        r1 = subprocess.run([os.path.join(EXEC, 'engine_linux64_gf'), '-i', f'{runname}_0001.rad', '-nt', str(nt)], cwd=work, capture_output=True, text=True, timeout=timeout, env=E)
    except subprocess.TimeoutExpired:
        p = Path(work, f'{runname}_0001.out')
        return (124, log0, (p.read_text() if p.exists() else '') + '\nENGINE TIMEOUT')
    out1 = os.path.join(work, f'{runname}_0001.out')
    log1 = open(out1).read() if os.path.exists(out1) else r1.stdout + r1.stderr
    return (r1.returncode, log0, log1)

def starter_errors(log):
    m = re.findall('(\\d+)\\s+ERROR\\(S\\)\\s*$', log, re.M)
    if m:
        return int(m[-1])
    m2 = re.search('(\\d+)\\s+ERRORS?\\s+FOUND', log)
    return int(m2.group(1)) if m2 else 0 if log.strip() else -1

def surface_closed(log):
    return 'UNABLE TO CLOSE EXTERNAL SURFACE' not in log

def initial_volume(log):
    m = re.search('INITIAL VOLUME OF MONITORED VOLUME[^=]*=\\s*([-\\dEe.+]+)', log, re.I)
    return float(m.group(1)) if m else None

def th_csv(work, runname):
    t01 = os.path.join(work, f'{runname}T01')
    if not os.path.exists(t01):
        raise FileNotFoundError(t01)
    subprocess.run([os.path.join(EXEC, 'th_to_csv_linux64_gf'), f'{runname}T01'], cwd=work, capture_output=True, text=True, env=env(), check=True)
    return t01 + '.csv'

def read_th(path):
    rows = [l.rstrip('\n').split(',') for l in open(path) if l.strip()]
    head = rows[0]
    data = np.array([[float(x) if x.strip() else np.nan for x in r] for r in rows[1:] if len(r) == len(head)], float)
    head = [h.strip().strip('"').strip() for h in head]
    (seen, cols) = ({}, {})
    for (k, h) in enumerate(head):
        seen[h] = seen.get(h, -1) + 1
        cols[h if seen[h] == 0 else f'{h}#{seen[h]}'] = data[:, k]
    return (data[:, 0], cols)

def cavity_work(cols, p_ext=0.0):
    if not any((k.split()[0] == 'monvol' for k in cols)):
        return 0.0
    mv = th_block(cols, 'monvol', 1, len(TH_MONV_VARS))[:, 0]
    V = mv[:, TH_MONV_VARS.index('VOL')]
    P = mv[:, TH_MONV_VARS.index('P')] - p_ext
    return np.concatenate([[0.0], np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(V))])

def patch_force(cols, delta, *, p_ext=0.0, n_jaw=2, win=9):
    w = np.asarray(cols['EXTERNAL WORK'], float) - cavity_work(cols, p_ext)
    d = np.asarray(delta, float)
    k = max(win // 2, 1)
    F = np.zeros_like(w)
    F[k:-k] = (w[2 * k:] - w[:-2 * k]) / np.where(np.abs(d[2 * k:] - d[:-2 * k]) > 1e-15, d[2 * k:] - d[:-2 * k], np.nan)
    (F[:k], F[-k:]) = (F[k], F[-k - 1])
    return F / n_jaw

def nodal_reaction(time, cols, n_nodes, *, tag='patch'):
    t = np.asarray(time, float)
    if len(t) < 3 or np.any(np.diff(t) <= 0):
        raise ValueError('reaction differentiation requires three ordered observations')
    names = [k for k in cols if k.split()[0] == tag]
    if not names or any((re.search('\\bvar\\s+\\d+$', k) is None for k in names)):
        raise ValueError('expected raw untitled impulse columns; inspect TH representation')
    a = th_block(cols, tag, n_nodes, len(TH_NODE_VARS))
    impulse = a[:, :, [TH_NODE_VARS.index(k) for k in ('REACX', 'REACY', 'REACZ')]]
    if np.any(~np.isfinite(impulse)):
        raise ValueError('nonfinite reaction impulse')
    return np.gradient(impulse, t, axis=0, edge_order=2)

def force_at_max(delta, F):
    i = np.where(np.isfinite(F))[0]
    return (float(delta[i[-1]]), float(F[i[-1]])) if len(i) else (np.nan, np.nan)

def th_block(cols, tag, n_obj, nvar):
    sel = [v for (k, v) in cols.items() if k.split()[0] == tag]
    if len(sel) != n_obj * nvar:
        raise RuntimeError(f'Unexpected time-history block size for {tag}: {len(sel)} != {n_obj}x{nvar}')
    return np.stack(sel, 1).reshape(-1, n_obj, nvar)

def last_cycle(log):
    row = None
    for line in log.splitlines():
        t = line.split()
        if len(t) == 13 and t[0].isdigit() and t[5].endswith('%'):
            row = t
    if row is None:
        return None
    f = lambda k: float(row[k])
    return dict(cycle=int(row[0]), t=f(1), dt=f(2), ie=f(6), ke=f(7) + f(8), ext=f(9), mass=f(11), added=f(12))

def mass_added(log):
    r = last_cycle(log)
    return None if r is None or r['mass'] <= 0 else 100.0 * r['added'] / r['mass']

def _vtk_scalar_max(path, keys=('EPSP', 'PLAS'), exact=False):
    (best, want, left) = (None, False, 0)

    def matches(name):
        return any((k.lower() == name.lower() if exact else k.lower() in name.lower() for k in keys))
    with open(path, errors='ignore') as fh:
        for line in fh:
            tok = line.split()
            if not tok:
                continue
            head = tok[0].upper()
            if head in ('SCALARS', 'VECTORS', 'NORMALS', 'TENSORS', 'POINT_DATA', 'CELL_DATA', 'FIELD', 'LOOKUP_TABLE', 'POINTS', 'CELLS', 'CELL_TYPES', 'POLYGONS', 'METADATA'):
                if head == 'SCALARS':
                    want = matches(tok[1])
                    left = -1
                elif head == 'LOOKUP_TABLE':
                    pass
                else:
                    want = False
                continue
            if not want and left <= 0:
                if len(tok) == 4 and tok[3].lower() in ('float', 'double') and matches(tok[0]):
                    try:
                        left = int(tok[1]) * int(tok[2])
                        want = True
                    except ValueError:
                        pass
                continue
            for x in tok:
                try:
                    v = float(x)
                except ValueError:
                    want = False
                    left = 0
                    break
                best = v if best is None else max(best, v)
                if left > 0:
                    left -= 1
                    if left == 0:
                        want = False
                        break
    return best

def _vtk_time(path):
    with open(path, errors='ignore') as fh:
        prev = ''
        for _ in range(16):
            line = fh.readline()
            if prev.startswith('TIME'):
                return float(line.split()[0])
            prev = line
    return None

def anim_epsp(work, runname, dt_anim=None, keys=('EPSP', 'PLAS'), exact=False, expected_layers=None):
    files = sorted(glob.glob(os.path.join(work, runname + 'A[0-9][0-9][0-9]')))
    E = env()
    (ts, mx) = ([], [])
    for (k, f) in enumerate(files):
        vtk = f + '.vtk'
        if not os.path.exists(vtk) or os.path.getsize(vtk) == 0:
            with open(vtk, 'w') as fh:
                subprocess.run([os.path.join(EXEC, 'anim_to_vtk_linux64_gf'), os.path.basename(f)], cwd=work, stdout=fh, stderr=subprocess.PIPE, env=E, timeout=600, check=True)
        if expected_layers is not None:
            values = [_vtk_scalar_max(vtk, (f'2DELEM_Plast_Strn_Layer_{j:3d}_'.replace(' ', '_'),), exact=True) for j in range(1, expected_layers + 1)]
            if any((v is None for v in values)):
                raise ValueError(f'incomplete thickness integration-point output: {f}')
            v = max(values)
        else:
            v = _vtk_scalar_max(vtk, keys, exact=exact)
        t = _vtk_time(vtk)
        os.remove(vtk)
        if v is None:
            raise ValueError(f'requested plastic strain field missing: {f}')
        if t is None:
            raise ValueError(f'animation observation time missing: {f}')
        ts.append(t)
        mx.append(v)
    return (np.asarray(ts, float), np.asarray(mx, float))

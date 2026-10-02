import os
import json
from pathlib import Path
import subprocess
import re
import numpy as np
from maniphysics.assessment.core.assessment import threshold_bounds
from maniphysics.assessment.core.quality import reaction_quality

def fortran_float(value):
    value = value.replace('D', 'E').replace('d', 'e')
    match = re.fullmatch('([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+))([+-]\\d{3,})', value)
    return float(match[1] + 'e' + match[2] if match else value)

def _deck_direction(stem, direction):
    d = np.asarray(direction, float)
    p = Path(stem + '.frame.json')
    if p.exists():
        d = np.asarray(json.loads(p.read_text())['world_to_deck']) @ d
    return d / np.linalg.norm(d)
OFFICIAL_CCX = Path(__file__).resolve().parents[2] / 'tools/calculix/2.23-official'
CCX = os.environ.get('CCX', str(OFFICIAL_CCX / 'ccx_2.23'))
EXTRA_ENVIRONMENT = {}

def runtime_files():
    if Path(CCX).resolve() == (OFFICIAL_CCX / 'ccx_2.23').resolve():
        return [OFFICIAL_CCX / 'libgfortran.so.4.0.0', OFFICIAL_CCX / 'manifest.json']
    return []

def validate_numeric_fields(path):
    number = re.compile('[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eEdD][+-]?\\d+)?')
    with Path(path).open() as stream:
        for (line_no, line) in enumerate(stream, 1):
            if line.lstrip().startswith('*'):
                continue
            for value in line.strip().split(','):
                value = value.strip()
                if len(value) > 20 and number.fullmatch(value):
                    raise ValueError(f'{path}:{line_no}: numeric field exceeds CCX 20-character input: {value}')

def ccx_environment(threads):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), CCX_NPROC_STIFFNESS=str(threads), CCX_NPROC_EQUATION_SOLVER=str(threads))
    env.update(EXTRA_ENVIRONMENT)
    if runtime_files():
        env['LD_LIBRARY_PATH'] = str(OFFICIAL_CCX) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return env

def run_ccx(job_dir, job, timeout=7200, threads=4):
    validate_numeric_fields(Path(job_dir) / (job + '.inp'))
    env = ccx_environment(threads)
    try:
        r = subprocess.run([CCX, job], cwd=job_dir, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or b''
        errors = exc.stderr or b''
        if isinstance(output, bytes):
            output = output.decode(errors='replace')
        if isinstance(errors, bytes):
            errors = errors.decode(errors='replace')
        Path(job_dir, job + '.solver.log').write_text(output + errors + '\nTIMEOUT\n')
        return (124, 'TIMEOUT')
    Path(job_dir, job + '.solver.log').write_text((r.stdout or '') + (r.stderr or ''))
    return (r.returncode, (r.stdout or '')[-4000:])

def converged(stem):
    f = stem + '.sta'
    if not os.path.exists(f):
        return False
    rows = [l for l in open(f) if l.strip() and (not l.lstrip().startswith('S'))]
    if not rows:
        return False
    try:
        return abs(float(rows[-1].split()[4]) - 1.0) < 1e-06
    except (ValueError, IndexError):
        return False

def blocks_with_time(dat, key, lo, hi, nset=None):
    (out, cur, take, t) = ([], None, True, None)
    with open(dat) as stream:
        for line in stream:
            low = line.lower()
            if key in low:
                take = nset is None or re.search('for set\\s+' + re.escape(nset.lower()) + '(?:\\s|$)', low) is not None
                mt = re.search('and time\\s+([-\\d.eE+]+)', line)
                t = float(mt.group(1)) if mt else None
                cur = [] if take else None
                if take:
                    out.append((t, cur))
                continue
            if cur is None:
                continue
            p = line.split()
            if len(p) < hi:
                if not line.strip() and cur:
                    cur = None
                continue
            try:
                cur.append([fortran_float(x) for x in p[lo:hi]])
            except ValueError:
                cur = None
    return [(t, b) for (t, b) in out if b]

def blocks_by_set(dat, key, lo, hi, nset):
    (out, cur, take) = ([], None, False)
    for line in open(dat):
        low = line.lower()
        if key in low:
            take = re.search('for set\\s+' + re.escape(nset.lower()) + '(?:\\s|$)', low) is not None
            cur = [] if take else None
            if take:
                out.append(cur)
            continue
        if cur is None:
            continue
        p = line.split()
        if len(p) < hi:
            if not line.strip() and cur:
                cur = None
            continue
        try:
            cur.append([fortran_float(x) for x in p[lo:hi]])
        except ValueError:
            cur = None
    return [b for b in out if b]

def blocks_with_time_multi(dat, requests):
    lookup = {(variable.lower(), nset.lower()): (name, columns) for (name, variable, nset, columns) in requests}
    if len(lookup) != len(requests):
        raise ValueError('output requests must have distinct variable/set pairs')
    output = {name: {} for (name, _, _, _) in requests}
    current = None
    columns = 0
    header = re.compile('for set\\s+(\\S+)\\s+and time\\s+(\\S+)', re.I)
    with open(dat) as stream:
        for line in stream:
            low = line.lower()
            match = header.search(low) if 'for set' in low else None
            if match:
                variable = low.split('(', 1)[0].strip()
                request = lookup.get((variable, match[1].lower()))
                current = None
                if request is not None:
                    (name, columns) = request
                    current = []
                    output[name][fortran_float(match[2])] = current
                continue
            if current is None:
                continue
            values = line.split()
            if len(values) < columns:
                if not line.strip() and current:
                    current = None
                continue
            try:
                current.append([fortran_float(x) for x in values[:columns]])
            except ValueError:
                current = None
    return {name: {t: block for (t, block) in times.items() if block} for (name, times) in output.items()}

def _blocks(dat, key, lo, hi):
    (out, cur) = ([], None)
    for line in open(dat):
        if key in line.lower():
            cur = []
            out.append(cur)
            continue
        if cur is None:
            continue
        p = line.split()
        if len(p) < hi:
            if not line.strip() and cur:
                cur = None
            continue
        try:
            cur.append([fortran_float(x) for x in p[lo:hi]])
        except ValueError:
            cur = None
    return [b for b in out if b]

def peeq_max(stem):
    B = _blocks(stem + '.dat', 'equivalent plastic strain', 2, 3)
    return None if not B else float(np.max(np.abs(np.array(B[-1]))))

def stress_summary(stem, block=-1):
    B = _blocks(stem + '.dat', 'stresses', 2, 8)
    if not B or abs(block) > len(B):
        return None
    S = np.array(B[block])
    T = np.zeros((len(S), 3, 3))
    (T[:, 0, 0], T[:, 1, 1], T[:, 2, 2]) = (S[:, 0], S[:, 1], S[:, 2])
    T[:, 0, 1] = T[:, 1, 0] = S[:, 3]
    T[:, 0, 2] = T[:, 2, 0] = S[:, 4]
    T[:, 1, 2] = T[:, 2, 1] = S[:, 5]
    ev = np.linalg.eigvalsh(T)
    dev = T - np.eye(3) * (np.trace(T, axis1=1, axis2=2) / 3.0)[:, None, None]
    vm = np.sqrt(1.5 * (dev ** 2).sum((1, 2)))
    return dict(s1_max=float(ev[:, 2].max()), s3_min=float(ev[:, 0].min()), vm_max=float(vm.max()), n_pt=len(S))

def drive_force(stem, direction):
    B = blocks_by_set(stem + '.dat', 'forces', 1, 4, 'NDRIVE')
    if not B:
        return None
    d = _deck_direction(stem, direction)
    return float(abs(np.array(B[-1]).sum(0) @ d))

def support_reactions(stem):
    meta = Path(stem + '.frame.json')
    if not meta.exists():
        return {}
    fixed = json.loads(meta.read_text()).get('fixed_dofs')
    if fixed is None:
        return {}
    result = {}
    for (t, rows) in blocks_with_time(stem + '.dat', 'forces', 0, 4, nset='NFIX'):
        by_node = {int(r[0]): np.asarray(r[1:]) for r in rows}
        if any((n not in by_node for (n, _) in fixed)):
            continue
        support = []
        for (n, dof) in fixed:
            vec = np.zeros(3)
            vec[dof - 1] = by_node[n][dof - 1]
            support.append(vec)
        result[t] = np.asarray(support)
    return result

def complete_field(block, expected):
    b = np.asarray(block, float)
    ids = np.asarray(expected['element_ids'], int)
    n = int(expected['points_per_element'])
    if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids) or (n <= 0):
        raise ValueError('nonempty unique requested elements and positive IP count required')
    if b.ndim != 2 or b.shape[1] < 3 or len(b) != len(ids) * n or (not np.isfinite(b).all()):
        return False
    pairs = b[np.lexsort((b[:, 1], b[:, 0])), :2]
    expected_pairs = np.column_stack((np.repeat(np.sort(ids), n), np.tile(np.arange(1, n + 1), len(ids))))
    return bool(np.array_equal(pairs, expected_pairs))

def ramp_series(stem, direction, crit, has_skin=False, *, expected_fields=None, expected_nodes=None):
    d = _deck_direction(stem, direction)
    text = Path(stem + '.inp').read_text()
    sets = re.findall('\\*EL PRINT,\\s*ELSET=([^,\\s]+)', text, re.I)
    read_set = sets[1] if crit == 'skin_s1' else sets[0]
    FB = blocks_with_time(stem + '.dat', 'forces', 0, 4, nset='NDRIVE')
    f = {t: abs(np.asarray(b)[:, 1:].sum(0) @ d) for (t, b) in FB}
    if crit == 'peeq':
        VB = blocks_with_time(stem + '.dat', 'equivalent plastic strain', 0, 3, nset=read_set)
        v = {t: float(np.max(np.asarray(b)[:, 2])) for (t, b) in VB}
    else:
        VB = blocks_with_time(stem + '.dat', 'stresses', 0, 8, nset=read_set)
        v = {}
        for (t, b) in VB:
            S = np.array(b)[:, 2:]
            if not np.isfinite(S).all():
                v[t] = np.nan
                continue
            T = np.zeros((len(S), 3, 3))
            (T[:, 0, 0], T[:, 1, 1], T[:, 2, 2]) = (S[:, 0], S[:, 1], S[:, 2])
            T[:, 0, 1] = T[:, 1, 0] = S[:, 3]
            T[:, 0, 2] = T[:, 2, 0] = S[:, 4]
            T[:, 1, 2] = T[:, 2, 1] = S[:, 5]
            ev = np.linalg.eigvalsh(T)
            v[t] = max(0.0, float(-ev[:, 0].min() if crit == 's3' else ev[:, 2].max()))
    if None in f or None in v:
        raise ValueError('increment time missing in solver output')
    if len(f) != len(FB) or len(v) != len(VB):
        raise ValueError('duplicate increment time in selected solver output set')
    t = np.array(sorted(set(f) | set(v)))
    result = dict(t=t, F=np.array([f.get(x, np.nan) for x in t]), V=np.array([v.get(x, np.nan) for x in t]))
    other = dict(blocks_with_time(stem + '.dat', 'forces', 0, 4, nset='NDRIVE_OTHER'))
    supports = support_reactions(stem)
    first = dict(FB)
    (balance, support_ratio, valid) = ([], [], [])
    for ti in t:
        if ti not in first or ti not in other or ti not in supports:
            balance.append(np.inf)
            support_ratio.append(np.inf)
            valid.append(False)
            continue
        (a, b, s) = (np.asarray(first[ti])[:, 1:].sum(0), np.asarray(other[ti])[:, 1:].sum(0), np.asarray(supports[ti]))
        q = reaction_quality(a, b, s, d)
        balance.append(q['force_balance'])
        support_ratio.append(q['support_ratio'])
        valid.append(q['valid'])
    result.update(force_balance=np.array(balance), support_ratio=np.array(support_ratio))
    result['valid'] = np.isfinite(result['F']) & np.isfinite(result['V']) & np.asarray(valid, bool)
    result['field_coverage_verified'] = expected_fields is not None
    if expected_fields is not None:
        coverage = {ti: complete_field(b, expected_fields[read_set]) for (ti, b) in VB}
        result['field_complete'] = np.array([coverage.get(ti, False) for ti in t])
        result['valid'] &= result['field_complete']
    result['reaction_coverage_verified'] = expected_nodes is not None
    if expected_nodes is not None:

        def complete_nodes(rows, name):
            actual = np.asarray(rows)[:, 0] if rows else np.array([])
            return np.array_equal(np.sort(actual), np.sort(expected_nodes[name]))
        result['reaction_complete'] = np.array([complete_nodes(first.get(ti, []), 'NDRIVE') and complete_nodes(other.get(ti, []), 'NDRIVE_OTHER') for ti in t])
        result['valid'] &= result['reaction_complete']
    result['reaction_relative_tolerance'] = 0.05
    result['reaction_absolute_tolerance_N'] = 1e-08
    return result

def ramp_curve(stem, direction, crit, has_skin=False):
    z = ramp_series(stem, direction, crit, has_skin)
    return (z['F'], z['V'])

def pad_curve(stem, d_in, crit='peeq', n_pad=2):
    B = _blocks(stem + '.dat', 'total force', 0, 3)
    if not B:
        return (np.array([]), np.array([]))
    R = np.array([b[0] for b in B]).reshape(-1, n_pad, 3)
    D = np.asarray(d_in, float).reshape(n_pad, 3)
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    F = np.abs((R * D).sum(2)).mean(1)
    if crit == 'peeq':
        PB = _blocks(stem + '.dat', 'equivalent plastic strain', 2, 3)
        V = np.array([np.max(np.abs(np.array(b))) for b in PB])
    else:
        SB = _blocks(stem + '.dat', 'stresses', 2, 8)
        V = []
        for b in SB:
            S = np.array(b)
            T = np.zeros((len(S), 3, 3))
            (T[:, 0, 0], T[:, 1, 1], T[:, 2, 2]) = (S[:, 0], S[:, 1], S[:, 2])
            T[:, 0, 1] = T[:, 1, 0] = S[:, 3]
            T[:, 0, 2] = T[:, 2, 0] = S[:, 4]
            T[:, 1, 2] = T[:, 2, 1] = S[:, 5]
            ev = np.linalg.eigvalsh(T)
            V.append(abs(ev[:, 0].min()) if crit == 's3' else ev[:, 2].max())
        V = np.array(V)
    n = min(len(F), len(V))
    return (F[:n], V[:n])

def threshold_from_curve(F, V, lim, max_extrap=0.0, *, linear=False):
    if max_extrap != 0:
        raise ValueError('nonlinear extrapolation is not a verified threshold')
    return threshold_bounds(F, V, lim, linear=linear).estimate

def reaction_check(stem, F_applied):
    B = list(support_reactions(stem).values())
    if not B:
        return None
    return float(np.abs(np.array(B[-1])).sum() / max(F_applied, 1e-12))

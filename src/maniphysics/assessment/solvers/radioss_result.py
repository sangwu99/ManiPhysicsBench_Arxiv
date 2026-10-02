import os
import json, subprocess, time
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get('MPB_RUNTIME_ROOT', 'runtime')).resolve()
from maniphysics.assessment.core import radioss as R
from maniphysics.assessment.core.quality import reaction_quality

def fields(path):
    s = path.read_text().splitlines()
    npoint = int(next((x.split()[1] for x in s if x.startswith('POINT_DATA'))))
    ncell = int(next((x.split()[1] for x in s if x.startswith('CELL_DATA'))))
    a = {}
    count = npoint
    for (i, l) in enumerate(s):
        if l.startswith('CELL_DATA'):
            count = ncell
        if l.startswith('SCALARS '):
            (_, name, *_) = l.split()
            j = i + 2
            v = []
            while len(v) < count:
                v.extend(map(float, s[j].split()))
                j += 1
            a[name] = np.asarray(v)
        elif l.startswith('VECTORS '):
            (_, name, *_) = l.split()
            j = i + 1
            v = []
            while len(v) < 3 * count:
                v.extend(map(float, s[j].split()))
                j += 1
            a[name] = np.asarray(v).reshape(count, 3)
    return (R._vtk_time(path), a)

def nodes(a, name, ids):
    n = a['NODE_ID'].astype(int)
    vals = []
    for i in ids:
        v = a[name][n == i]
        assert len(v) > 0, f'node {i} missing'
        assert np.allclose(v, v[0], rtol=1e-05, atol=1e-08), f'node {i} inconsistent copies'
        vals.append(v[0])
    return np.array(vals)

def analyze(out):
    start = time.monotonic()
    m = json.loads((out / 'input_audit.json').read_text())
    c = m['source_case']
    exe = json.loads((out / 'execution.json').read_text())
    rows = []
    for f in sorted(out.glob('modelA[0-9][0-9][0-9]')):
        v = out / 'direct_frame.tmp.vtk'
        with v.open('w') as log:
            subprocess.run([R.EXEC + '/anim_to_vtk_linux64_gf', f.name], cwd=out, env=R.env(), stdout=log, check=True)
        (t, a) = fields(v)
        v.unlink()
        rf = [nodes(a, 'Reaction_Forces', c['backs'][k]).sum(0) for k in (0, 1)]
        u = nodes(a, 'Displacement', c['centering_nodes'])
        support = -c['centering_N_m'] * u
        quality = reaction_quality(*rf, support, c['directions'][0], absolute_N=0.001)
        ids = a['ELEMENT_ID'].astype(int)
        mask = (ids >= 1) & (ids <= m['shell_elements'])
        assert mask.sum() == m['shell_elements']
        pe = max((float(a[f'2DELEM_Plast_Strn_Layer_{j:3d}_'.replace(' ', '_')][mask].max()) for j in range(1, m['nip'] + 1)))
        rows.append(dict(t=t, closing_fraction_of_ccx_full_ramp=t / m['duration_s'] * m['closing_fraction'], F=min((abs(np.array(c['directions'][0]) @ x) for x in rf)), peeq=pe, indicator_ratio=pe / c['limit'], jaw_reaction_N=[x.tolist() for x in rf], **quality))
    valid = []
    for r in rows:
        if not r['valid']:
            break
        valid.append(r)
    hit = next((i for (i, r) in enumerate(valid) if r['indicator_ratio'] >= 1), None)
    r = dict(status='unknown', rows=rows, valid_prefix_rows=len(valid), execution=exe, threshold=None, analysis_s=time.monotonic() - start)
    if hit is not None:
        a = valid[hit]
        b = valid[hit - 1] if hit else dict(F=0, closing_fraction_of_ccx_full_ramp=0)
        r.update(status='damaged', threshold=dict(force_at_damage_interval_N=[b['F'], a['F']], closing_fraction_interval=[b['closing_fraction_of_ccx_full_ramp'], a['closing_fraction_of_ccx_full_ramp']], force_monotonic_to_damage=bool(np.all(np.diff([x['F'] for x in valid[:hit + 1]]) >= -0.001))))
    elif exe['completed'] and len(valid) == len(rows) and valid:
        r['status'] = 'safe_in_calculated_range'
    (out / 'result.json').write_text(json.dumps(r, indent=2))
    return r

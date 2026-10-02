from pathlib import Path
import re
import numpy as np
VERSION = 's6-elastic-surface'

def stress_blocks(path, elset):
    rows = []
    time = None
    take = False
    with Path(path).open() as stream:
        for line in stream:
            if 'stresses (elem,' in line:
                if rows:
                    yield (time, np.asarray(rows))
                    rows = []
                take = re.search('for set\\s+' + re.escape(elset) + '(?:\\s|$)', line, re.I) is not None
                time = float(re.search('and time\\s+(\\S+)', line).group(1)) if take else None
            elif take:
                fields = line.split()
                if not fields:
                    if rows:
                        yield (time, np.asarray(rows))
                        rows = []
                        take = False
                elif len(fields) >= 8 and fields[0].isdigit() and fields[1].isdigit():
                    rows.append([float(x) for x in fields[:8]])
                else:
                    take = False
    if rows:
        yield (time, np.asarray(rows))

def invariants(s):
    tensor = np.zeros(s.shape[:-1] + (3, 3))
    (tensor[..., 0, 0], tensor[..., 1, 1], tensor[..., 2, 2]) = (s[..., 0], s[..., 1], s[..., 2])
    tensor[..., 0, 1] = tensor[..., 1, 0] = s[..., 3]
    tensor[..., 0, 2] = tensor[..., 2, 0] = s[..., 4]
    tensor[..., 1, 2] = tensor[..., 2, 1] = s[..., 5]
    ev = np.linalg.eigvalsh(tensor)
    vm = np.sqrt(0.5 * ((ev[..., 0] - ev[..., 1]) ** 2 + (ev[..., 1] - ev[..., 2]) ** 2 + (ev[..., 2] - ev[..., 0]) ** 2))
    return dict(s1_max=float(ev[..., -1].max()), s3_min=float(ev[..., 0].min()), vm_max=float(vm.max()))

def reconstruct(block):
    b = np.asarray(block, float)
    if b.ndim != 2 or b.shape[1] != 8 or (not len(b)) or (not np.isfinite(b).all()):
        raise ValueError('finite nonempty [element, IP, six stresses] rows required')
    b = b[np.lexsort((b[:, 1], b[:, 0]))]
    (ids, counts) = np.unique(b[:, 0], return_counts=True)
    if not np.all(counts == 9):
        raise ValueError('surface reconstruction requires single-layer S6 with 9 IPs')
    if not np.all(b[:, 1].reshape(-1, 9) == np.arange(1, 10)):
        raise ValueError('unexpected S6 integration-point numbering')
    s = b[:, 2:].reshape(len(ids), 3, 3, 6)
    z = np.sqrt(3 / 5)
    center = s.mean(axis=1)
    slope = (s[:, 2] - s[:, 0]) / (2 * z)
    surface = np.stack([center - slope, center + slope], axis=1)
    fit = np.stack([center - z * slope, center, center + z * slope], axis=1)
    residual = s - fit
    frobenius = np.sqrt((residual[..., :3] ** 2).sum(-1) + 2 * (residual[..., 3:] ** 2).sum(-1))
    ip = invariants(s)
    sf = invariants(surface)
    return dict(ip=ip, surface=sf, stress_fit_residual_Pa=float(frobenius.max()), n_elements=len(ids), n_ip=len(b), version=VERSION)

def series(stem, elset):
    inp = Path(str(stem) + '.inp').read_text()
    cards = re.findall('^(\\*EL PRINT[^\\n]*)\\n([^*]*)', inp, re.I | re.M)
    headers = [header for (header, data) in cards if re.search('ELSET\\s*=\\s*' + re.escape(elset) + '(?:\\s*,|\\s*$)', header, re.I) and 'S' in re.split('[\\s,]+', data.strip().upper())]
    if not headers or any(('GLOBAL=YES' not in line.upper().replace(' ', '') for line in headers)):
        raise ValueError('surface reconstruction requires explicit GLOBAL=YES stress output')
    rows = [dict(t=t, **reconstruct(b)) for (t, b) in stress_blocks(str(stem) + '.dat', elset)]
    if not rows:
        raise ValueError('no shell stress output for requested set')
    return rows

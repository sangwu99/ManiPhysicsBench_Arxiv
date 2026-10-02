def valid_prefix(result):
    if result.get('precontact_max_indicator_ratio', 0) >= 1:
        return ([], 'damage_before_bilateral_contact')
    rows = result.get('rows') or []
    n = result.get('evaluation_prefix_rows')
    t0 = result.get('evaluation_start_time_s')
    if n == 0:
        return ([], 'no_evaluation_prefix')
    if isinstance(n, int) and n > 0 and (t0 is not None):
        i0 = next((i for (i, r) in enumerate(rows) if (r.get('t') or 0) >= t0 - 1e-12), 0)
        seg = rows[i0:i0 + n]
        out = []
        for r in seg:
            if r.get('valid') is False:
                break
            out.append(r)
        return (out, None)
    if t0 is None:
        return ([], 'evaluation_start_missing')
    out = []
    for r in rows:
        if (r.get('t') or 0) < t0 - 1e-12:
            continue
        if r.get('valid') is False:
            break
        out.append(r)
    return (out, 'evaluation_prefix_length_missing')

def crossing_from(rows, allowance):
    prev = None
    for r in rows:
        (phi, f) = (r.get('plastic_volume_fraction'), r.get('F'))
        if phi is None or f is None:
            continue
        if phi >= allowance:
            if prev is None:
                return dict(kind='left_censored', lower=None, upper=float(f), interp=float(f), reason='damage_at_first_valid_row')
            (p0, f0) = prev
            if f < f0 - 1e-09:
                return dict(kind='path_dependent', lower=None, upper=None, interp=None, reason='force_decreases_across_allowance_crossing', observed_pair=[float(f0), float(f)])
            w = (allowance - p0) / (phi - p0) if phi != p0 else 0.0
            return dict(kind='bracketed', lower=float(f0), upper=float(f), interp=float(f0 + w * (f - f0)), phi_lo=float(p0), phi_hi=float(phi))
        if prev is not None and f < prev[1] - 1e-09:
            return dict(kind='path_dependent', lower=None, upper=None, interp=None, reason='force_decreases_before_allowance_crossing', observed_pair=[float(prev[1]), float(f)])
        prev = (phi, f)
    if prev is None:
        return dict(kind='failed', lower=None, upper=None, interp=None, reason='no_valid_row')
    return dict(kind='right_censored', lower=float(rows[-1]['F']), upper=None, interp=None, phi_max=float(max((r.get('plastic_volume_fraction') or 0 for r in rows))))

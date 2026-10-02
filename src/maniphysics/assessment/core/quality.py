import numpy as np

def reaction_quality(first, second, support, direction, *, relative=0.05, absolute_N=1e-08, physical_fixture=(0.0, 0.0, 0.0)):
    (a, b, s, d) = map(lambda x: np.asarray(x, float), (first, second, support, direction))
    fixture = np.asarray(physical_fixture, float)
    if a.shape != (3,) or b.shape != (3,) or s.ndim != 2 or (s.shape[1] != 3) or (d.shape != (3,)) or (fixture.shape != (3,)):
        raise ValueError('reaction vectors/support components have invalid shapes')
    d = d / np.linalg.norm(d)
    scale = max(abs(float(a @ d)), abs(float(b @ d)))
    residual = float(np.linalg.norm(a + b + s.sum(0) + fixture))
    auxiliary = float(np.linalg.norm(s, axis=1).sum())
    tolerance = absolute_N + relative * scale
    return dict(valid=bool(np.isfinite(residual) and np.isfinite(auxiliary) and (residual <= tolerance) and (auxiliary <= tolerance)), force_balance=residual / max(scale, absolute_N), support_ratio=auxiliary / max(scale, absolute_N), force_balance_residual_N=residual, support_force_N=auxiliary, reaction_tolerance_N=tolerance, physical_fixture_force_N=float(np.linalg.norm(fixture)))

def animation_curve(out, ramp_time, *, added_mass_percent, max_ke_ie=0.05, max_added_mass=2.0, max_jaw_imbalance=0.05, energy_resolution_J=0.0):
    if not np.isfinite(energy_resolution_J) or energy_resolution_J < 0:
        raise ValueError('energy resolution must be explicitly nonnegative in SI joules')
    t = np.asarray(out['t'], float)
    ta = np.asarray(out['peeq_t'], float)
    if np.any(np.diff(t) <= 0) or np.any(np.diff(ta) <= 0):
        raise ValueError('solver observation times must be strictly increasing')
    keep = ta <= ramp_time
    ta = ta[keep]
    v = np.asarray(out['peeq_frames'], float)[keep]
    f = np.asarray(out['F'], float).copy()
    zero = (t == 0) & (np.asarray(out['delta']) == 0) & (np.asarray(out['ie']) == 0) & (np.asarray(out['ke']) == 0)
    f[zero] = 0.0
    ratio = np.divide(out['ke'], out['ie'], out=np.full_like(t, np.inf), where=np.asarray(out['ie']) > 0)
    ratio[zero] = 0.0
    j = np.searchsorted(t, ta, side='left')
    within = (ta >= t[0]) & (ta <= t[-1])
    j = np.minimum(j, len(t) - 1)
    i = np.where(t[j] == ta, j, np.maximum(0, j - 1))
    (flo, fhi) = (np.minimum(f[i], f[j]), np.maximum(f[i], f[j]))
    imbalance = np.zeros_like(t)
    if 'F_A' in out and 'F_B' in out:
        (fa, fb) = (np.asarray(out['F_A']), np.asarray(out['F_B']))
        imbalance = np.divide(abs(fa - fb), 0.5 * (abs(fa) + abs(fb)), out=np.zeros_like(t), where=abs(fa) + abs(fb) > 0)
        imbalance[zero] = 0.0
    (ie, ke) = (np.asarray(out['ie'], float), np.asarray(out['ke'], float))
    energy_ok = np.isfinite(ie) & np.isfinite(ke) & (ie >= 0) & (ke >= 0) & (ke <= energy_resolution_J + max_ke_ie * ie)
    if energy_resolution_J == 0:
        energy_ok &= np.isfinite(ratio)
    good_th = np.isfinite(f) & (f >= 0) & energy_ok & np.isfinite(imbalance) & (imbalance <= max_jaw_imbalance)
    prefix = np.logical_and.accumulate(good_th)
    mass_ok = added_mass_percent is not None and added_mass_percent <= max_added_mass
    valid = within & prefix[j] & mass_ok & np.isfinite(v)
    return dict(t=ta, F=np.interp(ta, t, f), F_lower=flo, F_upper=fhi, V=v, delta=np.interp(ta, t, out['delta']), valid=valid, ke_ie=np.maximum(ratio[i], ratio[j]), kinetic_energy_J=np.maximum(ke[i], ke[j]), energy_excess_J=np.maximum((ke - max_ke_ie * ie)[i], (ke - max_ke_ie * ie)[j]), jaw_imbalance=np.maximum(imbalance[i], imbalance[j]), added_mass_percent=added_mass_percent, quality_limits=dict(ke_ie=max_ke_ie, added_mass_percent=max_added_mass, jaw_imbalance=max_jaw_imbalance, energy_resolution_J=energy_resolution_J))

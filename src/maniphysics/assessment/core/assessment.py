from dataclasses import asdict, dataclass
import math
import numpy as np
VERSION = 'abc-evidence-v3-complete-fields'

@dataclass(frozen=True)
class ThresholdBounds:
    kind: str
    lower: float | None = None
    upper: float | None = None
    estimate: float | None = None
    reason: str = ''
    completed: bool = False
    valid_points: int = 0
    crossing_index: int | None = None
    force_resolution_N: float = 0.0

    def to_dict(self):
        return asdict(self)

def threshold_bounds(F, V, lim, *, completed=False, valid=None, linear=False, force_lower=None, force_upper=None, force_resolution_N=0.0):
    (f, v) = (np.asarray(F, float), np.asarray(V, float))
    if f.ndim != 1 or v.shape != f.shape:
        raise ValueError('F and V must be matching one-dimensional arrays')
    if not math.isfinite(lim) or lim <= 0:
        raise ValueError('lim must be a positive, explicit detection threshold')
    if not math.isfinite(force_resolution_N) or force_resolution_N < 0:
        raise ValueError('force resolution must be finite and nonnegative')
    if linear and force_resolution_N:
        raise ValueError('linear force-resolution propagation is not defined here')
    good = np.isfinite(f) & np.isfinite(v) & (f >= 0) & (v >= 0)
    flo = f if force_lower is None else np.asarray(force_lower, float)
    fhi = f if force_upper is None else np.asarray(force_upper, float)
    if flo.shape != f.shape or fhi.shape != f.shape:
        raise ValueError('force uncertainty must match F')
    good &= np.isfinite(flo) & np.isfinite(fhi) & (flo >= 0) & (flo <= f) & (f <= fhi)
    flo = np.maximum(0.0, flo - force_resolution_N)
    fhi = fhi + force_resolution_N
    if valid is not None:
        mask = np.asarray(valid, bool)
        if mask.shape != f.shape:
            raise ValueError('valid must match F')
        good &= mask
    bad = np.flatnonzero(~good)
    n = int(bad[0]) if len(bad) else len(f)
    (f, v) = (f[:n], v[:n])
    (flo, fhi) = (flo[:n], fhi[:n])
    full = bool(completed and n == len(good))
    if not n:
        return ThresholdBounds('failed', reason='no_valid_prefix', completed=full)
    if linear:
        positive = (f > 0) & (v > 0)
        if not positive.any():
            return ThresholdBounds('failed', reason='no_nonzero_linear_response', valid_points=n)
        slopes = v[positive] / f[positive]
        if not np.allclose(slopes, slopes[0], rtol=1e-06, atol=0):
            raise ValueError('declared linear response is not proportional to force')
        threshold = float(lim / slopes[0])
        return ThresholdBounds('analytic_linear', threshold, threshold, threshold, 'declared_linear_model', full, n)
    hits = np.flatnonzero(v >= lim)
    i = int(hits[0]) if len(hits) else None
    prefix = f if i is None else f[:i + 1]
    if force_resolution_N:
        end = len(f) if i is None else i + 1
        descending = np.any(np.maximum.accumulate(flo[:end]) > fhi[:end])
    else:
        descending = np.any(np.diff(prefix) < -1e-08 * max(1.0, float(prefix.max())))
    if descending:
        return ThresholdBounds('path_dependent', reason='nonmonotone_force_requires_history', completed=full, valid_points=n, crossing_index=i, force_resolution_N=force_resolution_N)
    if i == 0:
        return ThresholdBounds('left_censored', lower=0.0, upper=float(fhi[0]), reason='first_valid_sample_damaged', completed=full, valid_points=n, crossing_index=0, force_resolution_N=force_resolution_N)
    if i is None:
        return ThresholdBounds('right_censored', lower=float(flo.max()), reason='no_damage_in_valid_prefix', completed=full, valid_points=n, force_resolution_N=force_resolution_N)
    (lo, hi) = (float(flo[i - 1]), float(fhi[i]))
    if f[i] <= f[i - 1]:
        return ThresholdBounds('path_dependent', reason='force_plateau_at_damage', completed=full, valid_points=n, crossing_index=i, force_resolution_N=force_resolution_N)
    estimate = float(f[i - 1] + (f[i] - f[i - 1]) * (lim - v[i - 1]) / (v[i] - v[i - 1]))
    return ThresholdBounds('bracketed', lo, hi, estimate, 'observed_crossing', full, n, i, force_resolution_N)

def classify_force(bounds, force, *, force_upper=None):
    b = bounds if isinstance(bounds, ThresholdBounds) else ThresholdBounds(**bounds)
    (lo, hi) = (float(force), float(force if force_upper is None else force_upper))
    if not (math.isfinite(lo) and math.isfinite(hi) and (0 <= lo <= hi)):
        raise ValueError('observed force interval must be finite and nonnegative')
    if b.kind in ('failed', 'path_dependent'):
        (status, reason) = ('unknown', b.reason)
    elif b.upper is not None and lo >= b.upper:
        (status, reason) = ('damaged', 'observed_force_reaches_damage_bound')
    elif b.lower is not None and hi <= b.lower:
        (status, reason) = ('safe', 'observed_force_within_verified_prefix')
    else:
        (status, reason) = ('unknown', 'force_inside_bracket_or_outside_coverage')
    return dict(status=status, damaged={'safe': False, 'damaged': True, 'unknown': None}[status], reason=reason, bounds=b.to_dict(), F_peak=hi, thr=b.estimate, criterion_version=VERSION)

def aggregate_events(events, *, coverage_reasons=(), history_verified=True):
    first = next((e for e in events if e['status'] == 'damaged'), None)
    reasons = list(coverage_reasons)
    if not history_verified:
        reasons.append('material_history_not_verified')
    if first is not None:
        (status, reason) = ('damaged', 'damage_in_at_least_one_grasp')
    elif not events or reasons or any((e['status'] == 'unknown' for e in events)):
        (status, reason) = ('unknown', 'incomplete_damage_evidence')
    else:
        (status, reason) = ('safe', 'all_assessed_grasps_safe')
    return dict(status=status, damaged={'safe': False, 'damaged': True, 'unknown': None}[status], reason=reason, events=events, coverage_reasons=reasons, first_violation=None if first is None else first.get('time_s'), criterion_version=VERSION)

def summarize(episodes):
    n = len(episodes)
    known_safe = sum((e.get('success') is True and e['status'] == 'safe' for e in episodes))
    possible_safe = sum((e.get('success') is not False and e['status'] != 'damaged' for e in episodes))
    return dict(n=n, success=sum((e.get('success') is True for e in episodes)), success_unknown=sum((e.get('success') is None for e in episodes)), damaged=sum((e['status'] == 'damaged' for e in episodes)), damage_unknown=sum((e['status'] == 'unknown' for e in episodes)), safe_success_count_lower=known_safe, safe_success_count_upper=possible_safe, safe_success_lower=known_safe / n if n else None, safe_success_upper=possible_safe / n if n else None, criterion_version=VERSION)

def scenario_envelope(scenarios, *, declared_scenarios):
    if not declared_scenarios or set(scenarios) != set(declared_scenarios):
        raise ValueError('all predeclared material/contact scenarios are required')
    indexed = {}
    for (name, rows) in scenarios.items():
        by_id = {row['path']: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError('duplicate episodes within a scenario')
        indexed[name] = by_id
    keys = set(indexed[declared_scenarios[0]])
    if any((set(rows) != keys for rows in indexed.values())):
        raise ValueError('scenario comparisons must preserve exactly the same cohort')
    episodes = []
    for path in sorted(keys):
        rows = [indexed[name][path] for name in declared_scenarios]
        if len({row.get('success') for row in rows}) != 1:
            raise ValueError('task-success labels must agree across scenarios')
        states = {row['status'] for row in rows}
        if not states <= {'safe', 'damaged', 'unknown'}:
            raise ValueError('invalid damage status')
        state = next(iter(states)) if len(states) == 1 else 'unknown'
        episodes.append(dict(path=path, success=rows[0].get('success'), status=state, scenario_statuses={name: indexed[name][path]['status'] for name in declared_scenarios}))
    return dict(episodes=episodes, summary=summarize(episodes), scenarios=list(declared_scenarios))

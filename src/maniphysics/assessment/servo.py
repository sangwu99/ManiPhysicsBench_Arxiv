import math
import numpy as np


def threshold(rows, *, kp_N_m=1000., actuator_limit_N=None, coupled=True):
    if not math.isfinite(kp_N_m) or kp_N_m <= 0:
        raise ValueError('kp_N_m must be positive')
    if actuator_limit_N is not None and (not math.isfinite(actuator_limit_N) or actuator_limit_N <= 0):
        raise ValueError('actuator_limit_N must be positive')
    base = dict(kind='failed', lower=None, upper=None, estimate=None, valid_points=0,
                force_convention='per_finger', coupled=coupled, kp_N_m=kp_N_m,
                actuator_limit_N=actuator_limit_N)
    previous = None
    maximum = 0.
    for i, row in enumerate(rows):
        if not row['valid']:
            return dict(base, kind='right_censored' if previous else 'failed',
                        lower=maximum if previous else None, reason=row.get('reason', 'invalid_state'), valid_points=i)
        f, q, indicator = (float(row[k]) for k in ('force_N', 'closure_m', 'indicator'))
        if not np.isfinite([f, q, indicator]).all() or min(f, q, indicator) < 0:
            raise ValueError('Invalid curve state')
        value = f + kp_N_m * q / 2 if coupled else f
        if previous is not None and q < previous[1] - 1e-10:
            return dict(base, kind='path_dependent', reason='loading_path_reverses', valid_points=i)
        if previous is not None and value < maximum - max(1e-3, .01 * maximum):
            return dict(base, kind='path_dependent', reason='unreachable_stable_branch', valid_points=i)
        if coupled and actuator_limit_N is not None and f > actuator_limit_N:
            if previous is not None and previous[3] < actuator_limit_N and previous[2] < 1:
                return dict(base, kind='right_censored', lower=previous[0],
                            reason='actuator_saturation_bracket', valid_points=i)
            return dict(base, reason='actuator_saturation_unresolved', valid_points=i)
        if indicator >= 1:
            if previous is None:
                return dict(base, kind='left_censored', upper=value,
                            reason='damage_at_first_valid_state', valid_points=i + 1)
            lo, _, ratio, _ = previous
            weight = (1 - ratio) / (indicator - ratio)
            return dict(base, kind='bracketed', lower=lo, upper=max(lo, value),
                        estimate=lo + weight * (value - lo), crossing_time_s=row['time_s'],
                        raw_reaction_N=f, total_compression_m=q, valid_points=i + 1)
        maximum = max(maximum, value)
        previous = (value, q, indicator, f)
    return dict(base, kind='right_censored' if previous else 'failed',
                lower=maximum if previous else None, reason='end_of_valid_path', valid_points=len(rows))


def classify(bounds, force_N):
    if not math.isfinite(force_N) or force_N < 0:
        raise ValueError('Observed force must be finite and nonnegative')
    if bounds['kind'] in ('failed', 'path_dependent'):
        return dict(status='unknown', reason=bounds.get('reason', bounds['kind']))
    cap = bounds.get('actuator_limit_N')
    if bounds.get('coupled') and cap is not None and force_N >= cap:
        if bounds.get('upper') is not None and bounds['upper'] < cap:
            return dict(status='damaged', reason='damage_precedes_actuator_saturation')
        return dict(status='unknown', reason='saturated_rigid_load_requires_closing_target')
    if bounds.get('upper') is not None and force_N > bounds['upper']:
        return dict(status='damaged', reason='exceeds_damage_upper_bound')
    if bounds.get('lower') is not None and force_N <= bounds['lower']:
        return dict(status='safe', reason='within_verified_undamaged_range')
    return dict(status='unknown', reason='threshold_interval_or_uncomputed_range')

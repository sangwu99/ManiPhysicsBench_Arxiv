import numpy as np

CONTACT_FLOOR_N = 0.001
DROP_REL = 0.01
FUNCTIONAL_WALL_CONTACT = 'opposite_wall_contact'
ADDED_MASS_MAX_PCT = 2.0

def _pad_contact_N(pad):
    v = pad.get('contact_N')
    return 0.0 if v is None else float(np.linalg.norm(np.asarray(v, float)))

def bilateral(frame):
    pads = frame.get('pads') or []
    return len(pads) >= 2 and all((_pad_contact_N(p) >= CONTACT_FLOOR_N for p in pads))

def _frame_gates(frame, energy=True):
    if not bilateral(frame):
        return 'bilateral_contact_lost'
    if energy and (not frame.get('relative_energy_small')):
        return 'relative_kinetic_energy'
    if not frame.get('object_dynamic_balance_valid'):
        return 'object_dynamic_balance'
    for p in frame.get('pads') or []:
        if not p.get('dynamic_balance_valid'):
            return 'pad_dynamic_balance'
        outside = p.get('outside_source_contact_surface_sum_force_magnitudes_N')
        if outside is not None and outside > 1e-06 + 0.001 * float(frame.get('contact_force_N') or 0):
            return 'interface_force_outside_source_contact_surface'
    return None

def _run_gates(audit, result, material_limit):
    bad = []
    added = (result or {}).get('added_mass_percent_of_original')
    if added is not None and added > ADDED_MASS_MAX_PCT:
        bad.append(f'added_mass_percent={added:.3g}>{ADDED_MASS_MAX_PCT}')
    ow = audit.get('onset_window') or {}
    if ow.get('samples') != 0 and ow.get('interface_force_surface_exclusive_passed') is False:
        bad.append('interface_force_not_surface_exclusive')
    if not audit.get('input_sha256_matches_source') in (None, True):
        bad.append('input_hash_mismatch')
    if material_limit is None:
        bad.append('material_domain_limit_missing')
    return bad


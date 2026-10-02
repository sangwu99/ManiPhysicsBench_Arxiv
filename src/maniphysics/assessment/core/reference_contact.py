from pathlib import Path
import hashlib
import json
import subprocess
import time
import numpy as np
from . import contact_history as ch, mesh3d, pad_contact as pc, run

def virgin_closing_motion(directions, displacement_m):
    zero = np.zeros((2, 3)).tolist()
    return dict(times_s=[0.0, 1.0, 2.0], translations_m=[zero, zero, (displacement_m * np.asarray(directions)).tolist()])

def prepare(directory, mid, points, directions, spec, *, motion=None, recorded_frames=None, numerics=None, source=None, pad=None, refinement_points=None):
    controls = {} if numerics is None else numerics
    pad = ch.Pad() if pad is None else pad
    points = np.asarray(points, float)
    directions = np.asarray(directions, float)
    if points.shape != (2, 3) or directions.shape != (2, 3):
        raise ValueError('two contacts and closing directions required')
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    if not all(((points.mean(0) - p) @ d > 0 for (p, d) in zip(points, directions))):
        raise ValueError('jaw directions must point into the object')
    if motion is not None and recorded_frames is not None:
        raise ValueError('supply motion or recorded frames, not both')
    virgin = motion is None and recorded_frames is None
    if virgin:
        motion = virgin_closing_motion(directions, spec['dmax'])
    h = controls.get('object_h_m', spec.get('contact_object_h_m', 0.002))
    moving = recorded_frames is not None or (not virgin and len(motion['times_s']) > 2)
    coarse = controls.get('object_coarse_h_m', h if moving else spec.get('contact_coarse_h_m', 0.004))
    mesh_id = spec.get('geometry_id', mid)
    refinement_points = points if refinement_points is None else np.asarray(refinement_points, float)
    built = mesh3d.build(mesh_id, spec['topo'], refinement_points, h_fine=h, h_coarse=coarse, r_in=controls.get('refinement_inner_radius_m', 1.25 * pad.radius_m), r_out=controls.get('refinement_outer_radius_m', 2 * pad.radius_m), quad=False, straight=True)
    (xyz, cells) = (built[0], built[1] - 1)
    skin = None
    if spec['topo'] == 'skin_core':
        skin = dict(elements=(built[3] - 1).tolist(), material=spec['skin'], thickness_m=spec['t'])
    (vp, ep) = ch.pad_mesh(pad, size_m=controls.get('pad_h_m', spec.get('contact_pad_h_m', 0.002)))
    pads = []
    registration = []
    for (point, direction) in zip(points, directions):
        surface = np.asarray(skin['elements']) if skin else cells
        thickness = spec.get('t') if skin or spec['topo'] == 'shell' else None
        (anchor, info) = ch.crown_support_plane(xyz, surface, point, direction, pad, shell_thickness_m=thickness)
        pads.append((pc.place_pad(vp, anchor, direction, gap=controls.get('initial_gap_m', spec.get('contact_initial_gap_m', 2e-06))), ep, direction))
        registration.append(info)
    evidence = dict(source or dict(kind='declared_reference_grasp'))
    if virgin:
        evidence['reference_initialization'] = dict(protocol='unloaded-first-step-v1', end_time_s=1.0, loading_time_offset_s=1.0, numerical_only=True)
    if recorded_frames is not None:
        raise ValueError('Recorded backing-pose replay is not supported by this API')
    evidence.update(model_id=mid, material_scenario=spec['assessment']['material'], initial_contacts_m=points.tolist(), inward_directions=directions.tolist(), mesh_controls=dict(object_h_m=h, object_coarse_h_m=coarse, pad_h_m=controls.get('pad_h_m', spec.get('contact_pad_h_m', 0.002))), geometry_id=mesh_id, registration=registration, material_evidence=spec['assessment'], refinement_points_m=refinement_points.tolist(), mechanical_input_mode='normal_force_and_tangential_pose' if 'normal_forces_N' in motion else 'declared_virgin_closing_ramp' if evidence['kind'] in ('grasp_force_curve', 'declared_reference_grasp') else 'actual_backing_pose_not_servo_target', material_domain=spec.get('material_domain', {}), mesh_array_sha256=hashlib.sha256(xyz.tobytes() + cells.tobytes()).hexdigest())
    return ch.write_case(directory, (xyz, cells), pads, material=spec['core'], pad=pad, motion=motion, criterion=spec['crit'], limit=spec['lim'], thickness_m=spec.get('t'), skin=skin, secondary_criterion=spec.get('crit2'), secondary_limit=spec.get('lim2'), centering_N_m=controls.get('centering_N_m', spec.get('centering_N_m', 0.01)), contact_penalty_Pa_m=controls.get('penalty_Pa_m', spec.get('contact_penalty_Pa_m', 200000000000.0)), friction=controls.get('friction', spec.get('contact_friction', 0.5)), max_increment_s=controls.get('max_increment_s', spec.get('ramp_inc', 0.02)), contact_slave=controls.get('contact_slave', spec.get('contact_slave', 'pad')), contact_type=controls.get('contact_type', spec.get('contact_type', 'SURFACE TO SURFACE')), pad_contact_surface=controls.get('pad_contact_surface', spec.get('pad_contact_surface', 'exposed')), source=evidence)

def run_virgin_prefix(out, *, threads, timeout):
    out = Path(out)
    stem = out / out.name
    run.validate_numeric_fields(stem.with_suffix('.inp'))
    started = time.monotonic()
    stamp = None
    reason = 'solver_exit'
    with stem.with_suffix('.solver.log').open('w') as log:
        process = subprocess.Popen([run.CCX, out.name], cwd=out, env=run.ccx_environment(threads), stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if time.monotonic() - started >= timeout:
                    reason = 'timeout'
                    break
                sta = stem.with_suffix('.sta')
                current = sta.stat().st_mtime_ns if sta.exists() else None
                if current is not None and current != stamp:
                    stamp = current
                    r = ch.analyze(out, 999, result_name='prefix_monitor.json')
                    if r['status'] == 'damaged':
                        reason = 'intentional_stop_after_verified_first_damage'
                        break
                time.sleep(5)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        log.write('\nREFERENCE RAMP TERMINATION: ' + reason + '\n')
    termination = dict(reason=reason, returncode=process.returncode, elapsed_s=time.monotonic() - started, requested_path_completed=False if reason != 'solver_exit' else None)
    (out / 'termination.json').write_text(json.dumps(termination, indent=2))
    return (process.returncode, termination)

def solve_curve(directory, mid, points, axis, spec, *, threads=2, timeout=7200):
    d = np.asarray(axis, float)
    d /= np.linalg.norm(d)
    chord = np.asarray(points, float)[1] - np.asarray(points, float)[0]
    chord /= np.linalg.norm(chord)
    if not np.isclose(abs(chord @ d), 1.0, rtol=0, atol=1e-10):
        raise ValueError('scalar reference curve requires the contact-chord axis')
    directions = np.array([d, -d])
    if (np.asarray(points)[1] - np.asarray(points)[0]) @ d < 0:
        directions = -directions
    prepare(directory, mid, points, directions, spec)
    out = Path(directory)
    (rc, termination) = run_virgin_prefix(out, threads=threads, timeout=timeout)
    result = ch.analyze(out, rc)
    rows = result['rows']
    data = dict(F=np.array([r.get('F', np.nan) for r in rows]), V=spec['lim'] * np.array([np.nan if r['indicator_ratio'] is None else r['indicator_ratio'] for r in rows]), peeq_max=np.array([r.get('peeq', np.nan) for r in rows]), V_primary=spec['lim'] * np.array([r.get('primary_indicator_ratio', np.nan) for r in rows]), V_secondary=(spec.get('lim2') or 1) * np.array([r.get('secondary_indicator_ratio', np.nan) for r in rows]), t=np.array([r['t'] for r in rows]), valid=np.array([r['valid'] for r in rows]), full=result['requested_path_completed'], solver_returncode=rc, contact_artifact=str(out / 'result.json'), spec_json=json.dumps(spec, sort_keys=True), criterion_version=ch.VERSION)
    data['termination_json'] = json.dumps(termination, sort_keys=True)
    return data

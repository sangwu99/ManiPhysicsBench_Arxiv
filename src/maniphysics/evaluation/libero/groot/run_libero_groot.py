from maniphysics.evaluation.record import finalize_episode
from maniphysics.paths import release_root, external_root
import argparse
import json
import math
import os
import sys
import time
import types
from pathlib import Path
import numpy as np
MPB = release_root()
CONTINUOUS = bool(os.environ.get('MANIPHYS_CONTINUOUS_GRIPPER'))
GRIP_FLIP = bool(os.environ.get('MANIPHYS_GROOT_GRIP_FLIP'))
DEFAULT_SUFFIXES = ['Walnut', 'Egg', 'CherryTomatoW30L40', 'Potato', 'Strawberry', 'Plum']
DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], np.float32)

def connect_n17(port, host):
    root = Path(os.environ.get('GR00T_REPO') or external_root() / 'repos' / 'Isaac-GR00T')
    sys.path.insert(0, str(root))
    pkg = types.ModuleType('gr00t.policy')
    pkg.__path__ = [str(root / 'gr00t' / 'policy')]
    sys.modules.setdefault('gr00t.policy', pkg)
    from gr00t.policy.server_client import PolicyClient
    return PolicyClient(host=host, port=port, timeout_ms=120000)

def quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / den

def obs_for_policy(obs, instruction, api):
    xyz = obs['robot0_eef_pos']
    rpy = quat2axisangle(obs['robot0_eef_quat'])
    img = np.ascontiguousarray(np.asarray(obs['agentview_image'], np.uint8)[::-1, ::-1])
    wri = np.ascontiguousarray(np.asarray(obs['robot0_eye_in_hand_image'], np.uint8)[::-1, ::-1])
    grip = np.asarray(obs['robot0_gripper_qpos'], np.float32)
    if api == 'n17':
        f = lambda v: np.asarray([[[float(v)]]], np.float32)
        (img, wri, grip) = (img[None, None], wri[None, None], grip[None, None])
    else:
        f = lambda v: np.asarray([[float(v)]], np.float32)
        (img, wri, grip) = (img[None], wri[None], grip[None])
    return {'video.image': img, 'video.wrist_image': wri, 'state.x': f(xyz[0]), 'state.y': f(xyz[1]), 'state.z': f(xyz[2]), 'state.roll': f(rpy[0]), 'state.pitch': f(rpy[1]), 'state.yaw': f(rpy[2]), 'state.gripper': grip, 'annotation.human.action.task_description': [instruction]}

def to_action(a, i, api):
    g = lambda k: float(np.ravel(a[k])[i])
    raw = g('action.gripper')
    lin = 1.0 - 2.0 * raw
    lin = -lin if GRIP_FLIP else lin
    grip = float(np.clip(lin, -1.0, 1.0)) if CONTINUOUS else float(np.sign(lin))
    return (np.array([g('action.x'), g('action.y'), g('action.z'), g('action.roll'), g('action.pitch'), g('action.yaw'), grip], dtype=np.float32), raw)

def rollout(env, client, instr, args):
    obs = env.reset()
    (grips, graws, steps, success, plan) = ([], [], 0, False, [])
    for _ in range(args.num_steps_wait):
        (obs, _r, _d, _i) = env.step(DUMMY_ACTION)
    while steps < args.max_steps and (not success):
        if not plan:
            act = client.get_action(obs_for_policy(obs, instr, args.api))
            act = act[0] if isinstance(act, tuple) else act
            n = min(args.replan_steps, len(np.ravel(act['action.x'])))
            plan = [to_action(act, i, args.api) for i in range(n)]
        (v, raw) = plan.pop(0)
        grips.append(float(v[6]))
        graws.append(float(raw))
        (obs, _r, _d, _i) = env.step(v)
        steps += 1
        success = bool(env.env._check_success())
    return (success, steps, grips, graws)
GRIP_BINS = [('le_-0.99', -1.01, -0.99), ('-0.99..-0.5', -0.99, -0.5), ('mid_-0.5..0.5', -0.5, 0.5), ('0.5..0.99', 0.5, 0.99), ('ge_0.99', 0.99, 1.01)]

def ep_row(ep, success, steps, grips, graws):
    g = np.asarray(grips)
    g01 = (g + 1.0) / 2.0
    hist = {name: float(((g >= lo) & (g < hi)).mean()) for (name, lo, hi) in GRIP_BINS}
    closing = g[g > 0]
    return dict(ep=ep, success=success, steps=steps, grip_min=float(g.min()), grip_max=float(g.max()), grip_unique=int(len(set(np.round(g, 3)))), grip_mid_frac=float(((g01 > 0.05) & (g01 < 0.95)).mean()), hist_frac=hist, mid_mass=hist['mid_-0.5..0.5'], close_sat=hist['ge_0.99'], open_sat=hist['le_-0.99'], partial_close=hist['0.5..0.99'], mean_close_cmd=float(closing.mean()) if closing.size else 0.0, mean_cmd=float(g.mean()), grip_raw_min=float(np.min(graws)), grip_raw_max=float(np.max(graws)), error=None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-slug', required=True)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=7202)
    ap.add_argument('--tasks', nargs='+', default=DEFAULT_SUFFIXES)
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--max-steps', type=int, default=600)
    ap.add_argument('--num-steps-wait', type=int, default=10)
    ap.add_argument('--replan-steps', type=int, default=8)
    ap.add_argument('--camera', type=int, default=256)
    ap.add_argument('--api', choices=['n17'], default='n17')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    from maniphysics.bench.libero.tasks import make_maniphys_env
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[libero] {args.model_slug} · tasks={args.tasks} · {args.episodes}ep · max_steps={args.max_steps} · replan={args.replan_steps}; gripper {('CONTINUOUS' if CONTINUOUS else 'BINARY (sign)')}", flush=True)
    client = connect_n17(args.port, args.host)
    results = {}
    for suffix in args.tasks:
        t0 = time.time()
        env = make_maniphys_env(suffix, log_dir=str(out / 'traj' / suffix), camera_size=args.camera)
        instr = env.language_instruction
        eps = []
        for ep in range(args.episodes):
            env.seed(ep)
            (suc, steps, grips, graws) = rollout(env, client, instr, args)
            finalize_episode(env, suc, ep, suffix)
            eps.append(ep_row(ep, suc, steps, grips, graws))
            print(f"  {suffix} ep{ep}: success={suc} steps={steps} grip∈[{eps[-1]['grip_min']:.2f},{eps[-1]['grip_max']:.2f}] unique={eps[-1]['grip_unique']} intermediate={eps[-1]['grip_mid_frac']:.2%}", flush=True)
        env.close()
        results[suffix] = dict(instruction=instr, episodes=eps)
        print(f"[{suffix}] successes {sum((e['success'] for e in eps))}/{len(eps)} ({time.time() - t0:.0f}s)", flush=True)
        (out / 'raw_rollout.json').write_text(json.dumps(dict(args=vars(args) | {'continuous': CONTINUOUS, 'resize': args.camera}, results=results), ensure_ascii=False, indent=1), encoding='utf-8')
    tot = sum((e['success'] for r in results.values() for e in r['episodes']))
    n = sum((len(r['episodes']) for r in results.values()))
    print(f'[libero] total successes {tot}/{n} → {out}', flush=True)
if __name__ == '__main__':
    main()

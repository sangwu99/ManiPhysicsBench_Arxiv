from maniphysics.evaluation.record import finalize_episode
from maniphysics.paths import release_root, external_root
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
MPB = release_root()
CONTINUOUS = bool(os.environ.get('MANIPHYS_CONTINUOUS_GRIPPER'))
STARVLA_FAMILY = {'starvla': ('starVLA', 'maniphysics.evaluation.clients.starvla', 'ModelClient'), 'm1': ('InternVLA-M1', 'maniphysics.evaluation.clients.m1', 'M1Inference'), 'vlajepa': ('VLA-JEPA', 'maniphysics.evaluation.clients.vlajepa', 'ModelClient')}
REMOTE_CLIENTS = {'remote', 'pizero', 'cogact'}
IMAGE_CLIENTS = {*STARVLA_FAMILY, *REMOTE_CLIENTS}
PROPRIO_CLIENTS = {'pizero'}

def make_env(env_id, max_steps):
    import gymnasium as gym
    __import__('maniphysics.bench.sapien.food')
    __import__('maniphysics.bench.sapien.containers')
    return gym.make(env_id, obs_mode='rgbd', prepackaged_config=True, max_episode_steps=max_steps)

def connect(port, host='localhost', kind='gr00t', ckpt=None):
    if kind in REMOTE_CLIENTS:
        from maniphysics.evaluation.policy_server import RemoteClient
        return RemoteClient(host, port)
    if kind in STARVLA_FAMILY:
        (repo, mod, cls_name) = STARVLA_FAMILY[kind]
        sys.path.insert(0, str(external_root() / 'repos' / repo))
        import importlib
        import websockets.sync.client as _wsc
        if not getattr(_wsc.connect, '_maniphys_keepalive', False):
            _orig_connect = _wsc.connect

            def _connect(*a, **k):
                k['ping_timeout'] = None
                k['open_timeout'] = max(k.get('open_timeout') or 0, 120)
                return _orig_connect(*a, **k)
            _connect._maniphys_keepalive = True
            _wsc.connect = _connect
        cls = getattr(importlib.import_module(mod), cls_name)
        return cls(policy_ckpt_path=ckpt, policy_setup='widowx_bridge', host=host, port=port)
    import types
    root = Path(os.environ.get('GR00T_REPO') or external_root() / 'repos' / 'Isaac-GR00T')
    sys.path.insert(0, str(root))
    pkg = types.ModuleType('gr00t.policy')
    pkg.__path__ = [str(root / 'gr00t' / 'policy')]
    sys.modules.setdefault('gr00t.policy', pkg)
    from gr00t.policy.server_client import PolicyClient
    return PolicyClient(host=host, port=port, timeout_ms=120000)

def eef_pos_from_env(env):
    import numpy as np
    import transforms3d.quaternions as tq
    u = env.unwrapped
    base = u.agent.base_pose.to_transformation_matrix()
    ee = u.tcp.pose.to_transformation_matrix()
    ee_in_base = np.linalg.inv(base) @ ee
    return np.concatenate([ee_in_base[:3, 3], tq.mat2quat(ee_in_base[:3, :3]), [1.0 - float(u.agent.get_gripper_closedness())]])

def obs_for_policy(env, obs, instruction, img_hw=(256, 256)):
    import cv2
    import transforms3d.euler as te
    import transforms3d.quaternions as tq
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    img = get_image_from_maniskill2_obs_dict(env.unwrapped, obs)
    p = eef_pos_from_env(env)
    default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    rpy = te.mat2euler(tq.quat2mat(p[3:7]) @ default_rot.T)
    f = lambda v: np.asarray([[[float(v)]]], np.float32)
    return {'video.image_0': cv2.resize(np.asarray(img[..., :3], np.uint8), (img_hw[1], img_hw[0]))[None, None], 'state.x': f(p[0]), 'state.y': f(p[1]), 'state.z': f(p[2]), 'state.roll': f(rpy[0]), 'state.pitch': f(rpy[1]), 'state.yaw': f(rpy[2]), 'state.pad': f(0.0), 'state.gripper': f(p[7]), 'annotation.human.action.task_description': [instruction]}

def to_action(a, i):
    g = lambda k: float(np.ravel(a[k])[i])
    grip = g('action.gripper')
    grip = float(np.clip(2.0 * grip - 1.0, -1.0, 1.0)) if CONTINUOUS else 2.0 * (grip > 0.5) - 1.0
    return np.array([g('action.x'), g('action.y'), g('action.z'), g('action.roll'), g('action.pitch'), g('action.yaw'), grip], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--port', type=int, default=7100)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--max-episode-steps', type=int, default=120)
    ap.add_argument('--chunk', type=int, default=8)
    ap.add_argument('--client', choices=['gr00t', *STARVLA_FAMILY, *REMOTE_CLIENTS], default='gr00t')
    ap.add_argument('--ckpt', default=None)
    a = ap.parse_args()
    out = Path(a.out_dir or MPB / 'outputs' / f'policyeval_{a.env}')
    (out / 'videos').mkdir(parents=True, exist_ok=True)
    print(f"[eval] {a.env} · {a.episodes} episodes; gripper {('CONTINUOUS' if CONTINUOUS else 'BINARY(stock)')}", flush=True)
    os.environ['MANIPHYS_LOG_DIR'] = str(out / 'traj' / a.env)
    client = connect(a.port, kind=a.client, ckpt=a.ckpt)
    env = make_env(a.env, a.max_episode_steps)
    rows = []
    for ep in range(a.episodes):
        (obs, _) = env.reset(seed=ep)
        instr = env.unwrapped.get_language_instruction()
        done = trunc = False
        (steps, grips) = (0, [])
        if a.client in IMAGE_CLIENTS:
            client.reset(instr)
        while not (done or trunc) and steps < a.max_episode_steps:
            if a.client in IMAGE_CLIENTS:
                from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
                img = np.asarray(get_image_from_maniskill2_obs_dict(env.unwrapped, obs)[..., :3], np.uint8)
                extra = {'eef_pos': eef_pos_from_env(env)} if a.client in PROPRIO_CLIENTS else {}
                (_raw, act) = client.step(img, instr, **extra)
                v = np.concatenate([np.ravel(act['world_vector']), np.ravel(act['rot_axangle']), np.ravel(act['gripper'])]).astype(np.float32)
                grips.append(float(v[6]))
                (obs, _, done, trunc, info) = env.step(v)
                steps += 1
                continue
            act = client.get_action(obs_for_policy(env, obs, instr))
            act = act[0] if isinstance(act, tuple) else act
            n = min(a.chunk, len(np.ravel(act['action.x'])))
            for i in range(n):
                v = to_action(act, i)
                grips.append(float(v[6]))
                (obs, _, done, trunc, info) = env.step(v)
                steps += 1
                if done or trunc or steps >= a.max_episode_steps:
                    break
        suc = bool(info.get('success', False))
        finalize_episode(env, suc, ep, a.env)
        g01 = (np.asarray(grips) + 1.0) / 2.0 if grips else np.zeros(0)
        rows.append(dict(episode=ep, success=suc, steps=steps, grip_min=min(grips) if grips else None, grip_max=max(grips) if grips else None, grip_unique=len(set(np.round(grips, 3))), grip_mid_frac=float(((g01 > 0.05) & (g01 < 0.95)).mean()) if grips else None))
        print(f"  ep{ep}: success={suc} steps={steps} grip∈[{rows[-1]['grip_min']:.2f},{rows[-1]['grip_max']:.2f}] unique {rows[-1]['grip_unique']}", flush=True)
    env.close()
    (out / 'summary.json').write_text(json.dumps(dict(env=a.env, continuous=CONTINUOUS, rows=rows, success_rate=sum((r['success'] for r in rows)) / max(len(rows), 1)), ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"[eval] successes {sum((r['success'] for r in rows))}/{len(rows)} → {out}")
if __name__ == '__main__':
    main()

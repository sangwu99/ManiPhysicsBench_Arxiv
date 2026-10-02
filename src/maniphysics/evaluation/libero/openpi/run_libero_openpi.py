from maniphysics.evaluation.record import finalize_episode
from maniphysics.paths import release_root
import argparse
import collections
import json
import math
import time
import traceback
from pathlib import Path
import numpy as np
MPB = release_root()
DUMMY_ACTION = [0.0] * 6 + [-1.0]
DEFAULT_SUFFIXES = ['Walnut', 'Egg', 'CherryTomatoW30L40', 'Potato', 'Strawberry', 'Plum']

def quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat = quat.copy()
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat = quat.copy()
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / den

def make_client(host, port):
    import functools
    import websockets.sync.client as _wsc
    from openpi_client import websocket_client_policy
    _wsc.connect = functools.partial(_wsc.connect, ping_timeout=None, open_timeout=120)
    return websocket_client_policy.WebsocketClientPolicy(host, port)

def build_element(obs, lang, resize_size, image_tools):
    img = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
    wrist = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    state = np.concatenate((obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']))
    return ({'observation/image': img, 'observation/wrist_image': wrist, 'observation/state': state, 'prompt': str(lang)}, img)

def run_suffix(suffix, args, client, image_tools, out_dir):
    from maniphysics.bench.libero.tasks import make_control_env, make_maniphys_env
    traj_dir = out_dir / 'traj' / suffix
    traj_dir.mkdir(parents=True, exist_ok=True)
    make = make_control_env if args.control else make_maniphys_env
    env = make(suffix, log_dir=str(traj_dir), camera_size=args.camera)
    lang = env.language_instruction
    print(f'[{suffix}] instruction={lang!r} gripper={type(env.env.robots[0].gripper).__name__}', flush=True)
    eps = []
    for ep in range(args.episodes):
        t0 = time.time()
        env.seed(ep)
        obs = env.reset()
        plan = collections.deque()
        (grips, frames) = ([], [])
        (success, steps, err) = (False, 0, None)
        try:
            for t in range(args.max_steps):
                if t < args.num_steps_wait:
                    (obs, _, done, _) = env.step(DUMMY_ACTION)
                    continue
                (element, img) = build_element(obs, lang, args.resize, image_tools)
                if args.video and t % 2 == 0:
                    frames.append(img)
                if not plan:
                    chunk = client.infer(element)['actions']
                    plan.extend(np.asarray(chunk)[:args.replan_steps])
                action = plan.popleft()
                grips.append(float(action[6]))
                (obs, _, done, _) = env.step(np.asarray(action, dtype=np.float64).tolist())
                steps = t + 1
                if done:
                    success = True
                    break
        except Exception as e:
            err = f'{type(e).__name__}: {e}'
            traceback.print_exc()
        g = np.asarray(grips) if grips else np.zeros(1)
        finalize_episode(env, success, ep, suffix, err)
        eps.append(dict(ep=ep, success=bool(success), steps=int(steps), error=err, secs=round(time.time() - t0, 1), grips=[round(float(x), 5) for x in grips], grip_n=int(g.size), grip_unique=int(np.unique(np.round(g, 4)).size), grip_min=round(float(g.min()), 4), grip_max=round(float(g.max()), 4), grip_mid_frac=round(float(np.mean(np.abs(g) < 0.99)), 4)))
        print(f"  [{suffix}] ep{ep:02d} success={success} steps={steps} grip[uniq={eps[-1]['grip_unique']} mid={eps[-1]['grip_mid_frac']} min={eps[-1]['grip_min']} max={eps[-1]['grip_max']}] {eps[-1]['secs']}s{('' if err is None else ' ERR ' + err)}", flush=True)
        if args.video and frames:
            import imageio.v2 as imageio
            vdir = out_dir / 'videos' / suffix
            vdir.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(str(vdir / f"ep{ep:02d}_{('S' if success else 'F')}.mp4"), [np.asarray(f, np.uint8) for f in frames], fps=10, macro_block_size=1)
    env.close()
    return dict(instruction=lang, episodes=eps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-slug', required=True)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--tasks', nargs='+', default=DEFAULT_SUFFIXES)
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--max-steps', type=int, default=600)
    ap.add_argument('--num-steps-wait', type=int, default=10)
    ap.add_argument('--replan-steps', type=int, default=5)
    ap.add_argument('--resize', type=int, default=224)
    ap.add_argument('--camera', type=int, default=256)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--control', action='store_true')
    args = ap.parse_args()
    from openpi_client import image_tools
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = make_client(args.host, args.port)
    print(f'[openpi] server metadata: {client.get_server_metadata()}', flush=True)
    raw = dict(model_slug=args.model_slug, args=vars(args), results={})
    for suffix in args.tasks:
        raw['results'][suffix] = run_suffix(suffix, args, client, image_tools, out_dir)
        (out_dir / 'raw_rollout.json').write_text(json.dumps(raw, indent=1, ensure_ascii=False))
        n = len(raw['results'][suffix]['episodes'])
        s = sum((e['success'] for e in raw['results'][suffix]['episodes']))
        print(f'[{suffix}] success {s}/{n}', flush=True)
    print(f'[openpi] DONE — {out_dir}', flush=True)
if __name__ == '__main__':
    main()

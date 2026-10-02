from maniphysics.evaluation.record import finalize_episode
from maniphysics.paths import release_root
import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
import numpy as np
MPB = release_root()
sys.path.insert(0, str(MPB))
from maniphysics.evaluation.libero.smolvla import rpc
DUMMY_ACTION = [0.0] * 6 + [-1.0]
DEFAULT_SUFFIXES = ['Walnut', 'Egg', 'CherryTomatoW30L40', 'Potato', 'Strawberry', 'Plum']

def quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64)
    w = min(1.0, max(-1.0, float(quat[3])))
    den = math.sqrt(max(0.0, 1.0 - w * w))
    if den < 1e-10:
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(w) / den

def build_obs(obs):
    return {'agentview_image': np.asarray(obs['agentview_image'], np.uint8), 'robot0_eye_in_hand_image': np.asarray(obs['robot0_eye_in_hand_image'], np.uint8), 'state': np.concatenate((obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos'])).astype(np.float32)}

def rollout(env, lang, client, args, ep, frames_out=None):
    env.seed(ep)
    obs = env.reset()
    client.reset(lang, episode=ep)
    (grips, success, steps, err) = ([], False, 0, None)
    try:
        for t in range(args.max_steps):
            if t < args.num_steps_wait:
                (obs, _, done, _) = env.step(DUMMY_ACTION)
                continue
            if frames_out is not None and t % 2 == 0:
                frames_out.append(np.asarray(obs['agentview_image'], np.uint8)[::-1, ::-1])
            action = client.infer(build_obs(obs))['actions']
            action = np.asarray(action, dtype=np.float64).reshape(-1)[:7]
            grips.append(float(action[6]))
            (obs, _, done, _) = env.step(action.tolist())
            steps = t + 1
            if done or env.env._check_success():
                success = True
                break
    except Exception as e:
        err = f'{type(e).__name__}: {e}'
        traceback.print_exc()
    return (success, steps, grips, err)

def _grip_stats(grips):
    g = np.asarray(grips, dtype=float) if grips else np.zeros(0)
    g01 = (g + 1.0) / 2.0
    return dict(grip_n=int(g.size), grip_min=float(g.min()) if g.size else None, grip_max=float(g.max()) if g.size else None, grip_unique=int(len(set(np.round(g, 3)))) if g.size else 0, grip_mid_frac=float(((g01 > 0.05) & (g01 < 0.95)).mean()) if g.size else None, grip_vals=[round(float(v), 5) for v in g])

def run_maniphys(suffix, args, client, out_dir):
    from maniphysics.bench.libero.tasks import make_maniphys_env
    traj_dir = out_dir / 'traj' / suffix
    traj_dir.mkdir(parents=True, exist_ok=True)
    env = make_maniphys_env(suffix, log_dir=str(traj_dir), camera_size=args.camera)
    lang = env.language_instruction
    print(f'[{suffix}] instruction={lang!r} gripper={type(env.env.robots[0].gripper).__name__}', flush=True)
    rows = []
    for ep in range(args.episodes):
        t0 = time.time()
        frames = [] if args.video else None
        (success, steps, grips, err) = rollout(env, lang, client, args, ep, frames)
        finalize_episode(env, success, ep, suffix, err)
        rows.append(dict(episode=ep, success=bool(success), steps=int(steps), error=err, secs=round(time.time() - t0, 1), **_grip_stats(grips)))
        print(f"  [{suffix}] ep{ep:02d} success={success} steps={steps} grip[uniq={rows[-1]['grip_unique']} mid={100 * (rows[-1]['grip_mid_frac'] or 0):.1f}% min={rows[-1]['grip_min']} max={rows[-1]['grip_max']}] {rows[-1]['secs']}s{('' if err is None else ' ERR ' + err)}", flush=True)
        if frames:
            import imageio.v2 as imageio
            vdir = out_dir / 'videos' / suffix
            vdir.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(str(vdir / f"ep{ep:02d}_{('S' if success else 'F')}.mp4"), frames, fps=10, macro_block_size=1)
    env.close()
    return dict(env=suffix, instruction=lang, continuous=bool(os.environ.get('MANIPHYS_CONTINUOUS_GRIPPER')), rows=rows, success=sum((r['success'] for r in rows)), n=len(rows))

def run_standard(suite_name, task_id, args, client):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=args.camera, camera_widths=args.camera)
    init_states = suite.get_task_init_states(task_id)
    lang = task.language
    print(f'[{suite_name}/{task_id}] {lang!r} gripper={type(env.env.robots[0].gripper).__name__}', flush=True)
    eps = []
    for ep in range(args.episodes):
        t0 = time.time()
        env.seed(ep)
        env.reset()
        env.set_init_state(init_states[ep % len(init_states)])
        client.reset(lang, episode=ep)
        (obs, grips, success, steps, err) = (None, [], False, 0, None)
        try:
            for t in range(args.max_steps):
                if t < args.num_steps_wait:
                    (obs, _, done, _) = env.step(DUMMY_ACTION)
                    continue
                action = np.asarray(client.infer(build_obs(obs))['actions'], dtype=np.float64).reshape(-1)[:7]
                grips.append(float(action[6]))
                (obs, _, done, _) = env.step(action.tolist())
                steps = t + 1
                if done or env.env._check_success():
                    success = True
                    break
        except Exception as e:
            err = f'{type(e).__name__}: {e}'
            traceback.print_exc()
        eps.append(dict(episode=ep, success=bool(success), steps=int(steps), error=err, secs=round(time.time() - t0, 1), **_grip_stats(grips)))
        print(f"  [{suite_name}/{task_id}] ep{ep:02d} success={success} steps={steps} {eps[-1]['secs']}s", flush=True)
    env.close()
    return dict(env=f'{suite_name}/{task_id}', instruction=lang, rows=eps, success=sum((r['success'] for r in eps)), n=len(eps))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=7404)
    ap.add_argument('--tasks', nargs='+', default=DEFAULT_SUFFIXES)
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--max-steps', type=int, default=600)
    ap.add_argument('--num-steps-wait', type=int, default=10)
    ap.add_argument('--replan-steps', type=int, default=1)
    ap.add_argument('--resize', type=int, default=512)
    ap.add_argument('--camera', type=int, default=256)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--standard-suite', default=None)
    ap.add_argument('--standard-task-ids', nargs='+', type=int, default=None)
    ap.add_argument('--model-slug', default='smolvla')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = rpc.PolicyClient(args.host, args.port)
    meta = client.meta()
    print(f'[{args.model_slug}] server meta: {meta}', flush=True)
    (summary_raw, out_raw) = ({}, out_dir / 'summary_raw.json')
    (out_dir / 'run_meta.json').write_text(json.dumps(dict(model_slug=args.model_slug, args=vars(args), server_meta=meta), indent=1, ensure_ascii=False), encoding='utf-8')
    if args.standard_suite:
        for tid in args.standard_task_ids or list(range(10)):
            rec = run_standard(args.standard_suite, tid, args, client)
            summary_raw[f'{args.standard_suite}/{tid}'] = rec
            out_raw.write_text(json.dumps(summary_raw, indent=1, ensure_ascii=False))
            print(f"[{args.standard_suite}/{tid}] success {rec['success']}/{rec['n']}", flush=True)
    else:
        for suffix in args.tasks:
            rec = run_maniphys(suffix, args, client, out_dir)
            summary_raw[suffix] = rec
            out_raw.write_text(json.dumps(summary_raw, indent=1, ensure_ascii=False))
            print(f"[{suffix}] success {rec['success']}/{rec['n']}", flush=True)
    tot_n = sum((r['n'] for r in summary_raw.values()))
    tot_s = sum((r['success'] for r in summary_raw.values()))
    print(f'[{args.model_slug}] DONE — {tot_s}/{tot_n} ({tot_s / max(tot_n, 1):.1%}) → {out_dir}', flush=True)
if __name__ == '__main__':
    main()

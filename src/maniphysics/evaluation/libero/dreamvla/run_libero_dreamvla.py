from maniphysics.evaluation.record import finalize_episode
from maniphysics.paths import release_root, external_root
import argparse
import json
import time
from pathlib import Path
import numpy as np
MPB = release_root()
DEFAULT_CKPT = external_root() / 'checkpoints/DreamVLA/libero_obj_94.pth'
DEFAULT_VIT = external_root() / 'checkpoints/DreamVLA/vit_mae/mae_pretrain_vit_base.pth'
DEFAULT_SUFFIXES = ['Walnut', 'Egg', 'CherryTomatoW30L40', 'Potato', 'Strawberry', 'Plum']

def kinematics(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    dist = np.linalg.norm(np.asarray(z['tcp_p']) - np.asarray(z['obj_p']), axis=1)
    hit = np.asarray(z['contact_impulse']) > 0
    n_contact = int(np.unique(np.asarray(z['control_step'])[hit]).size)
    return (n_contact, round(float(dist.min()), 4), bool(np.asarray(z['grasped']).max() > 0))

def grip_stats(g):
    g = np.asarray(g, dtype=float)
    close = g[g > 0.0]
    return dict(grip_n=int(g.size), grip_hist=[int((g <= -0.99).sum()), int(((g > -0.99) & (g <= -0.5)).sum()), int(((g > -0.5) & (g < 0.5)).sum()), int(((g >= 0.5) & (g < 0.99)).sum()), int((g >= 0.99).sum())], grip_mid_frac=round(float(np.mean(np.abs(g) < 0.5)), 4) if g.size else None, grip_close_n=int(close.size), grip_close_sat_n=int((close >= 0.99).sum()), grip_close_mean=round(float(close.mean()), 4) if close.size else None)
HIST_LABELS = ['<=-0.99', '-0.99~-0.5', '-0.5~+0.5', '+0.5~0.99', '>=0.99']

def run_task(suffix, policy, out_dir, episodes, camera_size, max_steps, video):
    from maniphysics.bench.libero.tasks import make_maniphys_env
    log_dir = out_dir / 'traj' / suffix
    log_dir.mkdir(parents=True, exist_ok=True)
    (per_ep, raw_gripper, cmd_gripper) = ([], [], [])
    for ep in range(episodes):
        env = make_maniphys_env(suffix, log_dir=str(log_dir), camera_size=camera_size)
        env.seed(ep)
        obs = env.reset()
        goal = env.language_instruction
        policy.reset()
        (success, frames) = (False, [])
        t0 = time.time()
        for step in range(max_steps):
            action = policy.step(obs, goal, step)
            (obs, reward, done, info) = env.step(action)
            if video:
                frames.append(np.asarray(obs['agentview_image'][::-1], np.uint8))
            if done:
                success = True
                break
        if not success:
            success = bool(env.env._check_success())
        path = finalize_episode(env, success, ep, suffix)
        env.close()
        (n_ct, mind, grasped) = kinematics(path)
        raw_gripper.extend(policy.raw_gripper_log)
        cmd_gripper.extend(policy.cmd_gripper_log)
        rec = dict(ep=ep, success=bool(success), trajectory=str(path.name), contact_steps=n_ct, min_tcp_obj_dist_m=mind, grasped=grasped, steps=step + 1, sec=round(time.time() - t0, 1))
        rec.update(grip_stats(policy.cmd_gripper_log))
        per_ep.append(rec)
        print(f'{suffix} ep{ep}: success={success}, contact_steps={n_ct}, steps={step + 1}', flush=True)
        if video and frames:
            import imageio.v2 as imageio
            vd = out_dir / 'videos' / suffix
            vd.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(str(vd / f'ep{ep}_s{int(success)}.mp4'), frames, fps=20, macro_block_size=1)
    n = len(per_ep)
    res = dict(success=sum((p['success'] for p in per_ep)), n=n, episodes_with_contact=sum((bool(p['contact_steps']) for p in per_ep)), per_episode=per_ep)
    return (res, raw_gripper, cmd_gripper)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=DEFAULT_SUFFIXES)
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--ckpt', default=str(DEFAULT_CKPT))
    ap.add_argument('--vit-ckpt', default=str(DEFAULT_VIT))
    ap.add_argument('--camera', type=int, default=128)
    ap.add_argument('--max-steps', type=int, default=600)
    ap.add_argument('--binary-gripper', action='store_true')
    ap.add_argument('--video', action='store_true')
    args = ap.parse_args()
    stamp = time.strftime('%m%d_%H%M%S')
    out = Path(args.out_dir or MPB / 'outputs' / f'libero_dreamvla_{stamp}')
    out.mkdir(parents=True, exist_ok=True)
    from maniphysics.evaluation.libero.dreamvla.policy import build_model, DreamVLAPolicy
    print(f'[dreamvla] loading {args.ckpt}', flush=True)
    (model, margs, load_info) = build_model(args.ckpt, args.vit_ckpt)
    print(f'[dreamvla] load_state_dict: {load_info}', flush=True)
    policy = DreamVLAPolicy(model, margs, continuous_gripper=not args.binary_gripper)
    summary = dict(model='dreamvla', checkpoint=f'WenyaoZhang/DreamVLA:{Path(args.ckpt).name}', gripper='binary' if args.binary_gripper else 'continuous', episodes_per_object=args.episodes, camera_size=args.camera, max_episode_steps=args.max_steps, results={})
    (all_raw, all_cmd) = ([], [])
    for suffix in args.tasks:
        (res, raw, cmd) = run_task(suffix, policy, out, args.episodes, args.camera, args.max_steps, args.video)
        summary['results'][suffix] = res
        all_raw.extend(raw)
        all_cmd.extend(cmd)
        print(f"[{suffix}] success {res['success']}/{res['n']}", flush=True)
        (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    if all_cmd:
        np.save(out / 'gripper_cmd.npy', np.asarray(all_cmd, float))
    if all_raw:
        a = np.asarray(all_raw, float)
        summary['gripper_raw_stats'] = dict(n=int(a.size), unique=int(np.unique(np.round(a, 6)).size), min=float(a.min()), max=float(a.max()), mean=float(a.mean()), hist_edges=[-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5], hist=np.histogram(a, bins=[-1000000000.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 1000000000.0])[0].tolist())
        np.save(out / 'gripper_raw.npy', a)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f'[dreamvla] DONE — {out}')
if __name__ == '__main__':
    main()

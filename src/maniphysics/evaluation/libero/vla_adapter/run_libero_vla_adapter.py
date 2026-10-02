from maniphysics.evaluation.record import finalize_episode
from maniphysics.evaluation.checkpoint import isolated_checkpoint
from maniphysics.paths import release_root, external_root
import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
import numpy as np
MPB = release_root()
DEFAULT_CKPT = str(external_root() / 'checkpoints/VLA-Adapter-LIBERO-Object-Pro')
UNNORM_KEY = 'libero_object_no_noops'
DEFAULT_SUFFIXES = ['Walnut', 'Egg', 'CherryTomatoW30L40', 'Potato', 'Strawberry', 'Plum']
if not any(('libero' in a.lower() for a in sys.argv)):
    sys.argv.append('--libero')

class Cfg:
    model_family = 'openvla'
    pretrained_checkpoint = DEFAULT_CKPT
    use_l1_regression = True
    use_minivlm = True
    use_film = False
    num_images_in_input = 2
    use_proprio = True
    center_crop = True
    num_open_loop_steps = 8
    unnorm_key = UNNORM_KEY
    load_in_8bit = False
    load_in_4bit = False
    save_version = 'vla-adapter'
    use_pro_version = True
    phase = 'Inference'

def build_policy(cfg):
    from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector
    from experiments.robot.robot_utils import get_model
    model = get_model(cfg)
    model.set_version(cfg.save_version)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = get_action_head(cfg, model.llm_dim)
    processor = get_processor(cfg)
    assert cfg.unnorm_key in model.norm_stats, f'unnorm_key {cfg.unnorm_key} not found; available keys: {list(model.norm_stats)[:5]}'
    return (model, action_head, proprio_projector, processor)

def prepare_observation(obs, resize_size):
    from experiments.robot.libero.libero_utils import get_libero_image, get_libero_wrist_image, quat2axisangle
    from experiments.robot.openvla_utils import resize_image_for_policy
    img = get_libero_image(obs)
    wrist = get_libero_wrist_image(obs)
    return ({'full_image': resize_image_for_policy(img, resize_size), 'wrist_image': resize_image_for_policy(wrist, resize_size), 'state': np.concatenate((obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']))}, img)

def process_action(action, binarize):
    from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action
    return invert_gripper_action(normalize_gripper_action(action, binarize=binarize))

def rollout(cfg, env, instr, policy, resize_size, max_steps, n_wait, binarize, frames=None):
    from experiments.robot.robot_utils import get_action
    (model, action_head, proprio_projector, processor) = policy
    obs = env.reset()
    queue = deque(maxlen=cfg.num_open_loop_steps)
    (grips, graws, steps, done) = ([], [], 0, False)
    for _ in range(n_wait):
        (obs, _, done, _) = env.step([0, 0, 0, 0, 0, 0, -1])
    while steps < max_steps and (not done):
        (observation, img) = prepare_observation(obs, resize_size)
        if frames is not None:
            frames.append(img)
        if len(queue) == 0:
            queue.extend(get_action(cfg, model, observation, instr, processor=processor, action_head=action_head, proprio_projector=proprio_projector, noisy_action_projector=None, use_film=cfg.use_film, use_minivlm=cfg.use_minivlm))
        raw = np.asarray(queue.popleft(), dtype=np.float64)
        graws.append(float(raw[6]))
        act = process_action(raw, binarize)
        grips.append(float(act[6]))
        (obs, _, done, _) = env.step(act.tolist())
        steps += 1
        if env.env._check_success():
            done = True
    return (bool(env.env._check_success()), steps, grips, graws)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--envs', nargs='*', default=DEFAULT_SUFFIXES)
    ap.add_argument('--n-ep', type=int, default=12)
    ap.add_argument('--max-steps', type=int, default=600)
    ap.add_argument('--camera-size', type=int, default=256)
    ap.add_argument('--num-steps-wait', type=int, default=10)
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--binarize-gripper', action='store_true')
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from maniphysics.evaluation.libero.run_libero import ep_row
    from maniphysics.bench.libero.tasks import make_maniphys_env
    from experiments.robot.robot_utils import get_image_resize_size
    cfg = Cfg()
    cfg.pretrained_checkpoint = isolated_checkpoint(a.ckpt, Path(a.out) / 'checkpoint_view')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cont = os.environ.get('MANIPHYS_CONTINUOUS_GRIPPER')
    print(f"[vla-adapter] ckpt={a.ckpt} envs={a.envs} n_ep={a.n_ep} max_steps={a.max_steps}; gripper {('CONTINUOUS' if cont else 'BINARY(stock)')} · binarize_action={a.binarize_gripper}", flush=True)
    policy = build_policy(cfg)
    resize_size = get_image_resize_size(cfg)
    summary = {}
    for suffix in a.envs:
        t0 = time.time()
        env = make_maniphys_env(suffix, log_dir=str(out / 'traj' / suffix), camera_size=a.camera_size)
        instr = env.language_instruction
        print(f'[{suffix}] instruction = {instr!r}', flush=True)
        rows = []
        for ep in range(a.n_ep):
            env.seed(ep)
            frames = [] if a.video else None
            (suc, steps, grips, graws) = rollout(cfg, env, instr, policy, resize_size, a.max_steps, a.num_steps_wait, a.binarize_gripper, frames)
            finalize_episode(env, suc, ep, suffix)
            rows.append(ep_row(ep, suc, steps, grips, graws, dump_grips=True))
            r = rows[-1]
            print(f"  {suffix} ep{ep}: success={suc} steps={steps} grip∈[{r['grip_min']:.2f},{r['grip_max']:.2f}] unique {r['grip_unique']} intermediate {100 * (r['grip_mid_frac'] or 0):.1f}% raw∈[{r['grip_raw_min']:.3f},{r['grip_raw_max']:.3f}] ({time.time() - t0:.0f}s)", flush=True)
            if frames:
                import imageio.v2 as imageio
                vd = out / 'videos' / suffix
                vd.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(str(vd / f'ep{ep}_success_{suc}.mp4'), [np.asarray(f, np.uint8) for f in frames], fps=20, macro_block_size=1)
        env.close()
        summary[suffix] = dict(env=suffix, instruction=instr, continuous=bool(cont), rows=rows, success=sum((r['success'] for r in rows)), n=len(rows))
        (out / 'summary_raw.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f"[{suffix}] successes {summary[suffix]['success']}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    tot = sum((v['success'] for v in summary.values()))
    n = sum((v['n'] for v in summary.values()))
    print(f'[vla-adapter] total successes {tot}/{n} → {out}')
if __name__ == '__main__':
    main()

from maniphysics.evaluation.record import finalize_episode
from maniphysics.evaluation.checkpoint import isolated_checkpoint
from maniphysics.paths import release_root
import argparse
import collections
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import numpy as np
MPB = release_root()
DUMMY_ACTION = [0.0] * 6 + [-1.0]
DEFAULT_SUFFIXES = ['Walnut', 'Egg', 'CherryTomatoW30L40', 'Potato', 'Strawberry', 'Plum']
CONTINUOUS = bool(os.environ.get('MANIPHYS_CONTINUOUS_GRIPPER'))

@dataclass
class Cfg:
    pretrained_checkpoint: Union[str, Path] = ''
    model_family: str = 'openvla'
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: str = ''
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    task_suite_name: str = 'libero_object'

def force_attn_implementation(attn):
    import types
    import experiments.robot.openvla_utils as ou
    orig = ou.AutoModelForVision2Seq

    def _from_pretrained(*a, **kw):
        kw.setdefault('attn_implementation', attn)
        return orig.from_pretrained(*a, **kw)
    ou.AutoModelForVision2Seq = types.SimpleNamespace(from_pretrained=_from_pretrained, register=orig.register)

def build_policy(args):
    from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector
    from experiments.robot.robot_utils import get_model, set_seed_everywhere
    if getattr(args, 'attn', None):
        force_attn_implementation(args.attn)
    set_seed_everywhere(args.seed)
    checkpoint = isolated_checkpoint(args.checkpoint, Path(args.out_dir) / 'checkpoint_view')
    cfg = Cfg(pretrained_checkpoint=checkpoint, task_suite_name=args.task_suite_name, num_open_loop_steps=args.chunk, lora_rank=args.lora_rank)
    vla = get_model(cfg)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    action_head = get_action_head(cfg, vla.llm_dim)
    processor = get_processor(cfg)
    key = cfg.task_suite_name
    if key not in vla.norm_stats and f'{key}_no_noops' in vla.norm_stats:
        key = f'{key}_no_noops'
    assert key in vla.norm_stats, f'unnorm_key {key!r} not in norm_stats: {list(vla.norm_stats)}'
    cfg.unnorm_key = key
    print(f'[oft] unnorm_key={key} · norm_stats keys={list(vla.norm_stats)}', flush=True)
    llm = getattr(vla, 'language_model', None)
    print(f"[oft] attn_implementation: top={vla.config._attn_implementation} llm={getattr(getattr(llm, 'config', None), '_attn_implementation', None)}", flush=True)
    if args.ript_lora:
        import torch
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(r=cfg.lora_rank, lora_alpha=min(cfg.lora_rank, 16), lora_dropout=0.0, target_modules='all-linear', init_lora_weights='gaussian')
        vla = get_peft_model(vla, lora_config)
        vla.load_adapter(args.ript_lora, adapter_name='default')
        header_path = os.path.join(args.ript_lora, 'openvla_headers.pt')
        head_sd = torch.load(header_path, map_location='cpu')['action_header']
        base_sd = {k: v.detach().float().clone() for (k, v) in action_head.state_dict().items()}
        action_head.load_state_dict(head_sd)
        after_sd = action_head.state_dict()
        head_changed = sum((1 for (k, v) in base_sd.items() if not torch.equal(v, after_sd[k].detach().float())))
        assert head_changed > 0, f'RIPT action head not loaded ({header_path})'
        head_moved = f'{head_changed}/{len(base_sd)} tensors changed'
        vla.eval()
        b_norm = sum((float(p.detach().float().norm()) for (n, p) in vla.named_parameters() if 'lora_B' in n))
        assert b_norm > 0, 'RIPT adapter not loaded: all LoRA B weights are zero'
        print(f'[ript] LoRA adapter {args.ript_lora} loaded; LoRA B norm sum={b_norm:.3f} · action_head {head_moved} · adapters={vla.active_adapters}', flush=True)
    return (cfg, vla, action_head, proprio_projector, processor)

def quat2axisangle(quat):
    import math
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / den

def prepare_observation(obs, resize_size):
    from experiments.robot.openvla_utils import resize_image_for_policy
    img = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
    wrist = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1])
    return ({'full_image': resize_image_for_policy(img, resize_size), 'wrist_image': resize_image_for_policy(wrist, resize_size), 'state': np.concatenate((obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']))}, img)

def grip_stats(g):
    g = np.asarray(g, dtype=float)
    close = g[g > 0.0]
    return dict(grip_n=int(g.size), grip_hist=[int((g <= -0.99).sum()), int(((g > -0.99) & (g <= -0.5)).sum()), int(((g > -0.5) & (g < 0.5)).sum()), int(((g >= 0.5) & (g < 0.99)).sum()), int((g >= 0.99).sum())], grip_mid_frac=round(float(np.mean(np.abs(g) < 0.5)), 4), grip_close_n=int(close.size), grip_close_sat_n=int((close >= 0.99).sum()), grip_close_mean=round(float(close.mean()), 4) if close.size else None, grip_min=round(float(g.min()), 4) if g.size else None, grip_max=round(float(g.max()), 4) if g.size else None)

def process_action(action, binarize):
    a = np.asarray(action, dtype=np.float64).copy()
    g = 2.0 * a[-1] - 1.0
    g = np.sign(g) if binarize else float(np.clip(g, -1.0, 1.0))
    a[-1] = -g
    return a

def run_suffix(suffix, args, policy, out_dir):
    import torch
    from maniphysics.bench.libero.tasks import make_maniphys_env
    from experiments.robot.robot_utils import get_action
    (cfg, vla, action_head, proprio_projector, processor) = policy
    traj_dir = out_dir / 'traj' / suffix
    traj_dir.mkdir(parents=True, exist_ok=True)
    env = make_maniphys_env(suffix, log_dir=str(traj_dir), camera_size=args.camera)
    lang = env.language_instruction
    print(f'[{suffix}] instruction={lang!r} gripper={type(env.env.robots[0].gripper).__name__}', flush=True)
    eps = []
    for ep in range(args.episodes):
        t0 = time.time()
        env.seed(ep)
        obs = env.reset()
        plan = collections.deque()
        (grips, raw_grips, frames) = ([], [], [])
        (success, steps, err) = (False, 0, None)
        try:
            for t in range(args.max_steps):
                if t < args.num_steps_wait:
                    (obs, _, _done, _) = env.step(DUMMY_ACTION)
                    continue
                (observation, img) = prepare_observation(obs, args.resize)
                if args.video and t % 2 == 0:
                    frames.append(img)
                if not plan:
                    chunk = get_action(cfg, vla, observation, lang, processor=processor, action_head=action_head, proprio_projector=proprio_projector, noisy_action_projector=None, use_film=cfg.use_film)
                    plan.extend(np.asarray(chunk)[:args.chunk])
                action = plan.popleft()
                raw_grips.append(float(action[6]))
                action = process_action(action, binarize=args.binarize_gripper)
                grips.append(float(action[6]))
                (obs, _, done, _) = env.step(action.tolist())
                steps = t + 1
                if done:
                    success = True
                    break
        except torch.cuda.OutOfMemoryError:
            print('[oft] CUDA OOM; evaluation aborted', flush=True)
            raise
        except Exception as e:
            err = f'{type(e).__name__}: {e}'
            traceback.print_exc()
        success = bool(success or env.check_success())
        ep_idx = int(env.env._mp_ep_idx)
        finalize_episode(env, success, ep, suffix, err)
        gs = grip_stats(grips if grips else [0.0])
        rg = np.asarray(raw_grips if raw_grips else [0.0], dtype=float)
        eps.append(dict(ep=ep, log_ep=ep_idx, success=success, steps=int(steps), error=err, secs=round(time.time() - t0, 1), **gs, raw_grip_min=round(float(rg.min()), 4), raw_grip_max=round(float(rg.max()), 4)))
        sat = eps[-1]['grip_close_sat_n'] / max(eps[-1]['grip_close_n'], 1)
        print(f"  [{suffix}] ep{ep:02d} success={success} steps={steps} grip[mid={eps[-1]['grip_mid_frac']} sat={sat:.3f} min={eps[-1]['grip_min']} max={eps[-1]['grip_max']}] {eps[-1]['secs']}s{('' if err is None else ' ERR ' + err)}", flush=True)
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
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--ript-lora', default=None)
    ap.add_argument('--task-suite-name', default='libero_object')
    ap.add_argument('--tasks', nargs='+', default=DEFAULT_SUFFIXES)
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--max-steps', type=int, default=600)
    ap.add_argument('--num-steps-wait', type=int, default=10)
    ap.add_argument('--chunk', type=int, default=8)
    ap.add_argument('--resize', type=int, default=224)
    ap.add_argument('--camera', type=int, default=256)
    ap.add_argument('--lora-rank', type=int, default=32)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--attn', default=None, choices=['flash_attention_2', 'sdpa', 'eager'])
    ap.add_argument('--binarize-gripper', action='store_true')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--video', action='store_true')
    args = ap.parse_args()
    if not args.binarize_gripper and (not CONTINUOUS):
        print('[warn] Continuous gripper disabled; using stock binary gripper', flush=True)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    from maniphysics.bench.libero.tasks import DISTRACTORS
    assert DISTRACTORS == [], f'Unexpected distractors={DISTRACTORS}'
    print(f'[oft] scene distractors={DISTRACTORS}', flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = build_policy(args)
    raw = dict(model_slug=args.model_slug, args=vars(args), gripper='binary' if args.binarize_gripper else 'continuous', env_continuous_switch=os.environ.get('MANIPHYS_CONTINUOUS_GRIPPER'), scene='nodist', distractors=list(DISTRACTORS), results={})
    for suffix in args.tasks:
        raw['results'][suffix] = run_suffix(suffix, args, policy, out_dir)
        (out_dir / 'raw_rollout.json').write_text(json.dumps(raw, indent=1, ensure_ascii=False))
        e = raw['results'][suffix]['episodes']
        print(f"[{suffix}] success {sum((x['success'] for x in e))}/{len(e)}", flush=True)
    print(f'[oft] DONE — {out_dir}', flush=True)
if __name__ == '__main__':
    main()

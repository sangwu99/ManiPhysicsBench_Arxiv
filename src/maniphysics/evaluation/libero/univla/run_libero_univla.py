from maniphysics.evaluation.record import finalize_episode
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from experiments.robot.libero.libero_utils import get_libero_image, quat2axisangle
from experiments.robot.libero.run_libero_eval import ActionDecoder
from experiments.robot.openvla_utils import get_vla_latent_action
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action
from maniphysics.bench.libero.tasks import FOOD_SUFFIXES, make_maniphys_env
LATENT_TOKENS = [f'<ACT_{i}>' for i in range(32)]
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
RESIZE_SIZE = 224
CAMERA_SIZE = 256
NUM_STEPS_WAIT = 10

class Cfg:

    def __init__(self, checkpoint, unnorm_key, center_crop):
        self.pretrained_checkpoint = str(checkpoint)
        self.unnorm_key = unnorm_key
        self.center_crop = center_crop

def load_univla(checkpoint, attn_impl):
    AutoConfig.register('openvla', OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    vla = AutoModelForVision2Seq.from_pretrained(checkpoint, attn_implementation=attn_impl, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to('cuda:0')
    vla.eval()
    stats_path = Path(checkpoint) / 'dataset_statistics.json'
    vla.norm_stats = json.loads(stats_path.read_text())
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    return (vla, processor)

def resolve_unnorm_key(norm_stats, want='libero_object'):
    if want in norm_stats:
        return want
    assert f'{want}_no_noops' in norm_stats, f'{want} not in norm_stats: {list(norm_stats)}'
    return f'{want}_no_noops'

def rollout(env, model, processor, decoder, cfg, instruction, max_steps, binarize, grip_log):
    decoder.reset()
    prev_hist_action = ['']
    replay = []
    stats = model.get_action_stats(cfg.unnorm_key)
    mask = np.array(stats.get('mask', np.ones_like(stats['q01'], dtype=bool)))
    (action_high, action_low) = (np.array(stats['q99']), np.array(stats['q01']))
    obs = env.env._get_observations()
    t = 0
    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            (obs, _, _, _) = env.step(DUMMY_ACTION)
            t += 1
            continue
        img = get_libero_image(obs, RESIZE_SIZE)
        replay.append(img)
        observation = {'full_image': img, 'state': np.concatenate((obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']))}
        (latent_action, visual_embed, generated_ids) = get_vla_latent_action(model, processor, cfg.pretrained_checkpoint, observation, instruction, cfg.unnorm_key, center_crop=cfg.center_crop, hist_action=prev_hist_action[-1])
        prev_hist_action.append(''.join((LATENT_TOKENS[i.item() - 32001] for i in generated_ids[0])))
        action = decoder(latent_action, visual_embed, mask, action_low, action_high)
        grip_log['decoder_raw'].append(float(action[-1]))
        action = normalize_gripper_action(action, binarize=binarize)
        action = invert_gripper_action(action)
        grip_log['env_cmd'].append(float(action[-1]))
        (obs, _, _, _) = env.step(action.tolist())
        t += 1
        if env.env._check_success():
            return (True, t, replay)
    return (False, t, replay)

def gripper_summary(values):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {}
    uniq = np.unique(np.round(v, 4))
    interior = float(np.mean((v > -0.95) & (v < 0.95)))
    return dict(n=int(v.size), n_unique_4dp=int(uniq.size), min=float(v.min()), max=float(v.max()), mean=float(v.mean()), std=float(v.std()), frac_interior=interior, hist_10bin=np.histogram(v, bins=10, range=(-1.0, 1.0))[0].tolist())
CONTACT_EPS = 1e-08
BINS = [(-np.inf, -0.99, 'le_-0.99'), (-0.99, -0.5, '-0.99..-0.5'), (-0.5, 0.5, 'mid_-0.5..0.5'), (0.5, 0.99, '0.5..0.99'), (0.99, np.inf, 'ge_0.99')]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--suffixes', nargs='*', default=FOOD_SUFFIXES)
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--max-steps', type=int, default=600)
    ap.add_argument('--window-size', type=int, default=12)
    ap.add_argument('--center-crop', type=int, default=1)
    ap.add_argument('--attn', default='flash_attention_2')
    ap.add_argument('--binarize-gripper', action='store_true')
    ap.add_argument('--save-video-eps', type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    (out / 'traj').mkdir(parents=True, exist_ok=True)
    (out / 'videos').mkdir(parents=True, exist_ok=True)
    (model, processor) = load_univla(args.checkpoint, args.attn)
    unnorm_key = resolve_unnorm_key(model.norm_stats)
    cfg = Cfg(args.checkpoint, unnorm_key, bool(args.center_crop))
    decoder = ActionDecoder(args.window_size)
    decoder.net.load_state_dict(torch.load(Path(args.checkpoint) / 'action_decoder.pt', map_location='cpu'))
    decoder.eval().cuda()
    print(f'[univla] unnorm_key={unnorm_key} window={args.window_size} binarize_gripper={args.binarize_gripper} attn={args.attn}', flush=True)
    summary = dict(model='univla', checkpoint='qwbu/univla-7b-224-sft-libero:univla-libero-object', gripper='binary' if args.binarize_gripper else 'continuous', episodes_per_object=args.episodes, max_episode_steps=args.max_steps, num_steps_wait=NUM_STEPS_WAIT, unnorm_key=unnorm_key, results={})
    all_grip = {'decoder_raw': [], 'env_cmd': []}
    per_obj_cmd = {}
    for suffix in args.suffixes:
        t0 = time.time()
        log_dir = out / 'traj'
        env = make_maniphys_env(suffix, log_dir=str(log_dir), camera_size=CAMERA_SIZE)
        instruction = env.language_instruction
        print(f"\n=== {suffix} | '{instruction}' ===", flush=True)
        grip_log = {'decoder_raw': [], 'env_cmd': []}
        (per_ep, successes) = ([], 0)
        for ep in range(args.episodes):
            env.seed(ep)
            env.reset()
            (ok, nsteps, replay) = rollout(env, model, processor, decoder, cfg, instruction, args.max_steps, args.binarize_gripper, grip_log)
            finalize_episode(env, ok, ep, suffix)
            successes += int(ok)
            per_ep.append(dict(ep=ep, success=ok, steps=nsteps))
            print(f'  ep{ep:02d} success={ok} steps={nsteps} ({successes}/{ep + 1}) {time.time() - t0:.0f}s', flush=True)
            if ep < args.save_video_eps and replay:
                import imageio
                imageio.mimwrite(out / 'videos' / f"{suffix}_ep{ep:02d}_{('S' if ok else 'F')}.mp4", np.stack(replay), fps=20)
        env.close()
        for k in all_grip:
            all_grip[k].extend(grip_log[k])
        summary['results'][suffix] = dict(success=successes, n=args.episodes, instruction=instruction, per_episode=per_ep, wall_sec=round(time.time() - t0, 1))
        per_obj_cmd[suffix] = list(grip_log['env_cmd'])
        (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    summary['gripper_overall'] = dict(decoder_raw=gripper_summary(all_grip['decoder_raw']), env_cmd=gripper_summary(all_grip['env_cmd']))
    np.savez_compressed(out / 'grips.npz', **{s: np.asarray(v, float) for (s, v) in per_obj_cmd.items()})
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print('\n=== DONE ===')
    print(json.dumps({k: {kk: v[kk] for kk in ('success', 'n')} for (k, v) in summary['results'].items()}, indent=1))
if __name__ == '__main__':
    main()

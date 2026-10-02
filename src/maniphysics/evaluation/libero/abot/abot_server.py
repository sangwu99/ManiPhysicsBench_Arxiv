from maniphysics.paths import release_root
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
MPB = release_root()
sys.path.insert(0, str(MPB))
from maniphysics.evaluation.libero.smolvla import rpc

def load_abot(ckpt_pt, base_vlm, device='cuda', bf16=True):
    from omegaconf import OmegaConf
    from ABot.model.framework import build_framework
    from ABot.model.tools import read_mode_config
    (model_config, norm_stats) = read_mode_config(Path(ckpt_pt))
    cfg = OmegaConf.create(model_config)
    cfg.trainer.pretrained_checkpoint = None
    cfg.framework.qwenvl.base_vlm = str(base_vlm)
    model = build_framework(cfg=cfg)
    model.norm_stats = norm_stats
    sd = torch.load(str(ckpt_pt), map_location='cpu', mmap=True, weights_only=True)
    emb_key = 'qwen_vl_interface.model.model.language_model.embed_tokens.weight'
    tgt_vocab = int(sd[emb_key].shape[0])
    cur_vocab = int(model.qwen_vl_interface.model.get_input_embeddings().weight.shape[0])
    if tgt_vocab != cur_vocab:
        print(f'[abot] vocab resize {cur_vocab} → {tgt_vocab}', flush=True)
        model.qwen_vl_interface.model.resize_token_embeddings(tgt_vocab)
    (missing, unexpected) = model.load_state_dict(sd, strict=False)
    hard_missing = [k for k in missing if 'lm_head' not in k]
    if hard_missing or unexpected:
        raise RuntimeError(f'state_dict mismatch: missing={hard_missing[:10]} ({len(hard_missing)} entries) unexpected={list(unexpected)[:10]} ({len(unexpected)} entries)')
    if missing:
        print(f'[abot] tied weight keys: {missing}', flush=True)
    if bf16:
        model = model.to(torch.bfloat16)
    return (model.to(device).eval(), norm_stats)

class ABotRunner:

    def __init__(self, ckpt_pt, base_vlm, unnorm_key='franka', device='cuda', chunk=None, image_size=224, continuous_gripper=True, seed=7):
        (self.model, self.norm_stats) = load_abot(ckpt_pt, base_vlm, device)
        self.stats = self.norm_stats[unnorm_key]['action']
        self.a_min = np.asarray(self.stats['min'], dtype=np.float64)
        self.a_max = np.asarray(self.stats['max'], dtype=np.float64)
        self.mask = np.asarray(self.stats.get('mask', [True] * 7), dtype=bool)
        cfg = self.model.config
        self.chunk = int(chunk or cfg.framework.action_model.future_action_window_size + 1)
        self.image_size = int(image_size)
        self.continuous = bool(continuous_gripper)
        self.seed = int(seed)
        (self.instruction, self._plan, self._step) = ('', None, 0)
        self.meta = dict(model='abot-m0', checkpoint=str(ckpt_pt), unnorm_key=unnorm_key, chunk=self.chunk, image_size=self.image_size, gripper='continuous' if self.continuous else 'binary(stock)', num_inference_timesteps=int(cfg.framework.action_model.num_inference_timesteps), action_dim=int(cfg.framework.action_model.action_dim), device=device)

    def _prep(self, arr):
        import cv2
        img = np.ascontiguousarray(np.asarray(arr, np.uint8)[::-1, ::-1])
        return cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

    def _unnormalize(self, n):
        n = np.clip(np.asarray(n, dtype=np.float64), -1.0, 1.0)
        g = n[:, 6].copy()
        if not self.continuous:
            g = np.where(g < 0.5, 0.0, 1.0)
        out = np.where(self.mask, 0.5 * (n + 1.0) * (self.a_max - self.a_min) + self.a_min, n)
        out[:, 6] = 1.0 - 2.0 * np.clip(g, 0.0, 1.0)
        return out

    def reset(self, instruction, episode=0):
        self.instruction = str(instruction)
        (self._plan, self._step) = (None, 0)
        torch.manual_seed(self.seed + episode)
        return dict(ok=True)

    def infer(self, obs):
        if self._step % self.chunk == 0:
            example = dict(image=[self._prep(obs['agentview_image']), self._prep(obs['robot0_eye_in_hand_image'])], lang=self.instruction)
            with torch.inference_mode():
                out = self.model.predict_action(examples=[example])
            n = np.asarray(out['normalized_actions'])[0]
            if n.shape[1] > 7:
                n = n[:, -7:]
            self._plan = self._unnormalize(n)
        action = self._plan[self._step % self.chunk]
        self._step += 1
        return dict(actions=action[None, :])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--base-vlm', required=True)
    ap.add_argument('--port', type=int, default=7504)
    ap.add_argument('--unnorm-key', default='franka')
    ap.add_argument('--chunk', type=int, default=None)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    cont = bool(os.environ.get('MANIPHYS_CONTINUOUS_GRIPPER'))
    t0 = time.time()
    runner = ABotRunner(a.ckpt, a.base_vlm, a.unnorm_key, chunk=a.chunk, image_size=a.image_size, continuous_gripper=cont, seed=a.seed)
    print(f'[abot] loaded in {time.time() - t0:.1f}s: {runner.meta}', flush=True)

    def handle(req):
        cmd = req.get('cmd')
        if cmd == 'meta':
            return dict(runner.meta)
        if cmd == 'reset':
            return runner.reset(req['instruction'], req.get('episode', 0))
        if cmd == 'infer':
            return runner.infer(req['obs'])
        return dict(error=f'unknown cmd {cmd!r}')
    rpc.serve(a.port, handle, tag='abot')
if __name__ == '__main__':
    main()

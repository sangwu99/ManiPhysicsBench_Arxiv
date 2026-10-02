from maniphysics.paths import release_root
import argparse
import sys
import time
import numpy as np
import torch
MPB = release_root()
sys.path.insert(0, str(MPB))
from maniphysics.evaluation.libero.smolvla import rpc

class SmolVLARunner:

    def __init__(self, ckpt, device='cuda', n_action_steps=None, num_steps=None, seed=7):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        self.policy = SmolVLAPolicy.from_pretrained(ckpt)
        cfg = self.policy.config
        if n_action_steps is not None:
            cfg.n_action_steps = int(n_action_steps)
        if num_steps is not None:
            cfg.num_steps = int(num_steps)
        (self.pre, self.post) = make_pre_post_processors(cfg, pretrained_path=str(ckpt))
        self.device = device
        self.policy.to(device)
        self.policy.eval()
        self.instruction = ''
        self.seed = int(seed)
        self.meta = dict(model='smolvla', checkpoint=str(ckpt), chunk_size=cfg.chunk_size, n_action_steps=cfg.n_action_steps, num_steps=cfg.num_steps, num_vlm_layers=cfg.num_vlm_layers, vlm_model_name=cfg.vlm_model_name, resize_imgs_with_padding=cfg.resize_imgs_with_padding, input_features=sorted(cfg.input_features), device=device, seed=self.seed)

    def _img(self, arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float().div_(255.0)
        return torch.flip(t.unsqueeze(0), dims=[2, 3])

    def reset(self, instruction, episode=0):
        self.instruction = str(instruction)
        self.policy.reset()
        torch.manual_seed(self.seed + int(episode))
        return dict(ok=True)

    def infer(self, obs):
        batch = {'observation.images.image': self._img(obs['agentview_image']), 'observation.images.image2': self._img(obs['robot0_eye_in_hand_image']), 'observation.state': torch.from_numpy(np.asarray(obs['state'], dtype=np.float32)).unsqueeze(0), 'task': [self.instruction]}
        batch = self.pre(batch)
        with torch.inference_mode():
            action = self.policy.select_action(batch)
        action = self.post(action)
        return dict(actions=action.to('cpu').float().numpy())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--port', type=int, default=7404)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n-action-steps', type=int, default=None)
    ap.add_argument('--num-steps', type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    runner = SmolVLARunner(a.ckpt, a.device, a.n_action_steps, a.num_steps)
    print(f'[smolvla] loaded in {time.time() - t0:.1f}s: {runner.meta}', flush=True)

    def handle(req):
        cmd = req.get('cmd')
        if cmd == 'meta':
            return dict(runner.meta)
        if cmd == 'reset':
            return runner.reset(req['instruction'], req.get('episode', 0))
        if cmd == 'infer':
            return runner.infer(req['obs'])
        return dict(error=f'unknown cmd {cmd!r}')
    rpc.serve(a.port, handle, tag='smolvla')
if __name__ == '__main__':
    main()

from maniphysics.paths import release_root, external_root
import argparse
import os
import socket
import sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
MPB = release_root()
PI0 = external_root() / 'repos' / 'open-pi-zero'
for p in (str(MPB), str(PI0), str(PI0 / '.deps')):
    sys.path.insert(0, p)
from maniphysics.evaluation.policy_server import _recv, _send
PALIGEMMA = Path.home() / '.cache/huggingface/hub/models--google--paligemma-3b-pt-224' / 'snapshots' / '35e4f46485b4d07967e7e9935bc3786aad50687c'

class PiZeroSimplerPolicy:

    def __init__(self, cfg, model, adapter, device, dtype):
        (self.cfg, self.model, self.adapter) = (cfg, model, adapter)
        (self.device, self.dtype) = (device, dtype)
        self.adapter.get_image = lambda env, obs: obs['_image']
        self.queue = []

    def reset(self, instruction=None):
        self.adapter.reset()
        self.queue = []

    def _predict(self, image, eef_pos, instruction):
        obs = {'_image': np.asarray(image, np.uint8), 'agent': {'eef_pos': np.asarray(eef_pos, np.float64)}}
        inp = self.adapter.preprocess(None, obs, instruction)
        (cm, vlm_pos, prop_pos, act_pos) = self.model.build_causal_mask_and_position_ids(inp['attention_mask'], dtype=self.dtype)
        (it_mask, act_mask) = self.model.split_full_mask_into_submasks(cm)
        kw = dict(input_ids=inp['input_ids'], pixel_values=inp['pixel_values'].to(self.dtype), image_text_proprio_mask=it_mask, action_mask=act_mask, vlm_position_ids=vlm_pos, proprio_position_ids=prop_pos, action_position_ids=act_pos, proprios=inp['proprios'].to(self.dtype))
        kw = {k: v.to(self.device) for (k, v) in kw.items()}
        with torch.inference_mode():
            raw = self.model(**kw)[0].float().cpu().numpy()
        env_actions = self.adapter.postprocess(raw)
        n = int(self.cfg.act_steps)
        self.queue = [(env_actions[i], float(raw[i, 6])) for i in range(n)]

    def step(self, image, instruction=None, eef_pos=None, *a, **kw):
        if not self.queue:
            self._predict(image, eef_pos, instruction)
        (v, raw_grip) = self.queue.pop(0)
        raw = {'world_vector': v[:3], 'rotation_delta': v[3:6], 'open_gripper': np.array([raw_grip])}
        action = {'world_vector': v[:3], 'rot_axangle': v[3:6], 'gripper': v[6:7], 'terminate_episode': np.array([0.0])}
        return (raw, action)

def build(cfg_path, ckpt, device, dtype, paligemma):
    import hydra
    from src.model.vla.pizero import PiZeroInference
    cfg = OmegaConf.load(cfg_path)
    cfg.env.adapter.pretrained_model_path = str(paligemma)
    cfg.env.adapter.dataset_statistics_path = str(PI0 / cfg.env.adapter.dataset_statistics_path)
    cfg.use_bf16 = dtype == torch.bfloat16
    model = PiZeroInference(cfg, use_ddp=False)
    data = torch.load(ckpt, weights_only=True, map_location='cpu')
    data['model'] = {k.replace('_orig_mod.', ''): v for (k, v) in data['model'].items()}
    model.load_state_dict(data['model'], strict=True)
    model.freeze_all_weights()
    model.to(dtype).to(device).eval()
    return (cfg, model, hydra.utils.instantiate(cfg.env.adapter))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--port', type=int, default=7205)
    ap.add_argument('--config', default=str(PI0 / 'config' / 'eval' / 'bridge.yaml'))
    ap.add_argument('--paligemma', default=str(PALIGEMMA))
    ap.add_argument('--use-bf16', action='store_true')
    a = ap.parse_args()
    device = torch.device('cuda:0')
    dtype = torch.bfloat16 if a.use_bf16 else torch.float32
    print(f"[pizero] loading: {a.ckpt}; gripper {('CONTINUOUS' if os.environ.get('MANIPHYS_CONTINUOUS_GRIPPER') else 'BINARY(stock)')}", flush=True)
    (cfg, model, adapter) = build(a.config, a.ckpt, device, dtype, a.paligemma)
    policy = PiZeroSimplerPolicy(cfg, model, adapter, device, dtype)
    policy.step(np.zeros((256, 256, 3), np.uint8), 'warmup', eef_pos=np.array([0.3, 0.0, 0.1, 0.0, 0.707, 0.0, 0.707, 1.0]))
    policy.reset()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', a.port))
    srv.listen(4)
    print(f'[pizero] ready on :{a.port} · act_steps={cfg.act_steps} · dtype={dtype}', flush=True)
    while True:
        (conn, _) = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            req = _recv(conn)
            if req is None:
                break
            if req['cmd'] == 'reset':
                policy.reset(req.get('instruction'))
                _send(conn, dict(ok=True))
            else:
                (raw, act) = policy.step(req['image'], req.get('instruction'), eef_pos=req['eef_pos'])
                _send(conn, dict(raw={k: np.asarray(v) for (k, v) in raw.items()}, action={k: np.asarray(v) for (k, v) in act.items()}))
        conn.close()
if __name__ == '__main__':
    main()

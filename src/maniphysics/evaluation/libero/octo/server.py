from maniphysics.paths import release_root, external_root
import argparse
import os
import socket
import sys
from collections import deque
import numpy as np
MPB = release_root()
sys.path.insert(0, str(external_root() / 'repos' / 'octo'))
sys.path.insert(0, str(MPB))
from maniphysics.evaluation.policy_server import _recv, _send

class ActionEnsembler:

    def __init__(self, horizon, temp=0.0):
        (self.horizon, self.temp) = (horizon, temp)
        self.hist = deque(maxlen=horizon)

    def reset(self):
        self.hist.clear()

    def ensemble(self, chunk):
        self.hist.append(np.asarray(chunk))
        n = len(self.hist)
        acts = np.stack([self.hist[i][n - 1 - i] for i in range(n)])
        w = np.exp(-self.temp * np.arange(n))
        return (acts * (w / w.sum())[:, None]).sum(0)

class OctoLiberoPolicy:

    def __init__(self, ckpt, step, unnorm_key, exec_horizon=4, ensemble_temp=None, seed=0, gripper_sign=1.0):
        import jax
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        from octo.model.octo_model import OctoModel
        (self.jax, self.tf) = (jax, tf)
        self.model = OctoModel.load_pretrained(str(ckpt), step)
        if unnorm_key not in self.model.dataset_statistics:
            raise KeyError(f"unnorm_key '{unnorm_key}' not found; available: {list(self.model.dataset_statistics)}")
        self.stats = self.model.dataset_statistics[unnorm_key]['action']
        self.img_size = int(self.model.example_batch['observation']['image_primary'].shape[-2])
        self.window = int(self.model.example_batch['observation']['image_primary'].shape[1])
        self.exec_horizon = exec_horizon
        self.ensembler = ActionEnsembler(exec_horizon, ensemble_temp) if ensemble_temp is not None else None
        self.gripper_sign = float(gripper_sign)
        self.continuous = bool(os.environ.get('MANIPHYS_CONTINUOUS_GRIPPER'))
        self.rng = jax.random.PRNGKey(seed)
        self.reset('warmup')

    def _resize(self, image):
        im = self.tf.image.resize(image, size=(self.img_size, self.img_size), method='lanczos3', antialias=True)
        return self.tf.cast(self.tf.clip_by_value(self.tf.round(im), 0, 255), self.tf.uint8).numpy()

    def reset(self, instruction):
        self.task = self.model.create_tasks(texts=[instruction])
        self.instruction = instruction
        self.hist = deque(maxlen=self.window)
        self.queue = deque()
        if self.ensembler:
            self.ensembler.reset()

    def _predict_chunk(self, image):
        self.hist.append(self._resize(image))
        while len(self.hist) < self.window:
            self.hist.appendleft(self.hist[0])
        imgs = np.stack(self.hist)[None]
        pad = np.ones((1, self.window), dtype=bool)
        (self.rng, key) = self.jax.random.split(self.rng)
        obs = {'image_primary': imgs, 'timestep_pad_mask': pad, 'pad_mask_dict': {'image_primary': pad, 'timestep': pad}}
        chunk = self.model.sample_actions(obs, self.task, unnormalization_statistics=self.stats, rng=key)
        return np.asarray(chunk[0], dtype=np.float64)

    def _to_libero(self, a):
        out = np.empty(7, dtype=np.float64)
        out[:6] = np.clip(a[:6], -1.0, 1.0)
        g = float(a[6])
        cont = np.clip(self.gripper_sign * (1.0 - 2.0 * g), -1.0, 1.0)
        out[6] = cont if self.continuous else 1.0 if cont > 0 else -1.0
        return (out, g)

    def step(self, image, instruction=None):
        if instruction and instruction != self.instruction:
            self.reset(instruction)
        if self.ensembler is not None:
            a = self.ensembler.ensemble(self._predict_chunk(image))
        else:
            if not self.queue:
                chunk = self._predict_chunk(image)
                self.queue.extend(chunk[:self.exec_horizon])
            a = self.queue.popleft()
        (act, g_raw) = self._to_libero(a)
        return (act, g_raw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--step', type=int, default=None)
    ap.add_argument('--unnorm-key', default='libero_object')
    ap.add_argument('--exec-horizon', type=int, default=4)
    ap.add_argument('--ensemble-temp', type=float, default=None)
    ap.add_argument('--gripper-sign', type=float, default=1.0)
    ap.add_argument('--port', type=int, default=7301)
    a = ap.parse_args()
    print(f"[octo] loading {a.ckpt} step={a.step} unnorm={a.unnorm_key}; gripper {('CONTINUOUS' if os.environ.get('MANIPHYS_CONTINUOUS_GRIPPER') else 'BINARY(stock)')}", flush=True)
    policy = OctoLiberoPolicy(a.ckpt, a.step, a.unnorm_key, exec_horizon=a.exec_horizon, ensemble_temp=a.ensemble_temp, gripper_sign=a.gripper_sign)
    policy.step(np.zeros((256, 256, 3), np.uint8), 'warmup')
    policy.reset('warmup')
    print(f'[octo-libero] img={policy.img_size} window={policy.window} exec_horizon={a.exec_horizon} ensemble={a.ensemble_temp} gripper_sign={a.gripper_sign:+.0f}', flush=True)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', a.port))
    srv.listen(4)
    print(f'[octo-libero] ready on :{a.port}', flush=True)
    while True:
        (conn, _) = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            req = _recv(conn)
            if req is None:
                break
            if req['cmd'] == 'reset':
                policy.reset(req['instruction'])
                _send(conn, dict(ok=True))
            else:
                (act, g_raw) = policy.step(req['image'], req.get('instruction'))
                _send(conn, dict(action=act, gripper_raw=g_raw))
        conn.close()
if __name__ == '__main__':
    main()

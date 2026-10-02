from maniphysics.paths import release_root, external_root
import argparse
import os
import pickle
import socket
import struct
import sys
import types
import numpy as np
MPB = release_root()
SIMPLER = external_root() / 'repos' / 'SimplerEnv'

def _send(sock, obj):
    b = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack('!Q', len(b)) + b)

def _recv(sock):
    buf = b''
    while len(buf) < 8:
        c = sock.recv(8 - len(buf))
        if not c:
            return None
        buf += c
    n = struct.unpack('!Q', buf)[0]
    body = b''
    while len(body) < n:
        c = sock.recv(min(1 << 20, n - len(body)))
        if not c:
            return None
        body += c
    return pickle.loads(body)

class RemoteClient:

    def __init__(self, host='localhost', port=7200):
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def reset(self, task_description):
        _send(self.sock, dict(cmd='reset', instruction=task_description))
        return _recv(self.sock)

    def step(self, image, task_description=None, *a, **kw):
        _send(self.sock, dict(cmd='step', image=np.asarray(image, np.uint8), instruction=task_description, **kw))
        r = _recv(self.sock)
        if r is None or 'error' in r:
            raise RuntimeError(f'Policy server error: {r}')
        return (r['raw'], r['action'])

def _stub_simpler_env():
    if 'simpler_env' in sys.modules:
        return
    pkg = types.ModuleType('simpler_env')
    pkg.__path__ = [str(SIMPLER / 'simpler_env')]
    sys.modules['simpler_env'] = pkg

def build_policy(model, ckpt):
    sys.path.insert(0, str(SIMPLER))
    _stub_simpler_env()
    if model == 'octo':
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        from octo.model.octo_model import OctoModel
        from maniphysics.evaluation.clients.octo import OctoInference
        return OctoInference(model=OctoModel.load_pretrained(str(ckpt)), policy_setup='widowx_bridge')
    if model == 'spatialvla':
        from maniphysics.evaluation.clients.spatialvla import SpatialVLAInference
        return SpatialVLAInference(saved_model_path=str(ckpt), policy_setup='widowx_bridge')
    raise ValueError(model)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['octo', 'spatialvla'], required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--port', type=int, default=7200)
    a = ap.parse_args()
    print(f"[server] {a.model} loading: {a.ckpt}; gripper {('CONTINUOUS' if os.environ.get('MANIPHYS_CONTINUOUS_GRIPPER') else 'BINARY(stock)')}", flush=True)
    policy = build_policy(a.model, a.ckpt)
    policy.reset('warmup')
    policy.step(np.zeros((256, 256, 3), np.uint8), 'warmup')
    policy.reset('warmup')
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', a.port))
    srv.listen(4)
    print(f'[server] ready on :{a.port}', flush=True)
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
                (raw, act) = policy.step(req['image'], req['instruction'])
                _send(conn, dict(raw={k: np.asarray(v) for (k, v) in raw.items()}, action={k: np.asarray(v) for (k, v) in act.items()}))
        conn.close()
if __name__ == '__main__':
    main()

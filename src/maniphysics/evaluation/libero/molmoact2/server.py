import argparse
import socket
import struct
import pickle
import time
import traceback
import numpy as np

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

class Policy:

    def __init__(self, ckpt, dtype='float32', num_steps=None, cuda_graph=True):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.dtype = getattr(torch, dtype)
        self.num_steps = num_steps
        self.cuda_graph = cuda_graph
        print(f'[molmoact2] loading {ckpt} (dtype={dtype})', flush=True)
        self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True, use_fast=False)
        self.model = AutoModelForImageTextToText.from_pretrained(ckpt, trust_remote_code=True, dtype=self.dtype, low_cpu_mem_usage=True).to('cuda').eval()
        print('[molmoact2] loaded', flush=True)

    def infer(self, images, task, state):
        from PIL import Image
        pil = [Image.fromarray(np.asarray(im, np.uint8)).convert('RGB') for im in images]
        kw = dict(processor=self.processor, images=pil, task=str(task), state=np.asarray(state, np.float32), norm_tag='libero', inference_action_mode='continuous', enable_depth_reasoning=False, normalize_language=True, enable_cuda_graph=self.cuda_graph)
        if self.num_steps is not None:
            kw['num_steps'] = self.num_steps
        torch = self.torch
        if self.dtype == torch.float32:
            with torch.inference_mode():
                out = self.model.predict_action(**kw)
        else:
            with torch.inference_mode(), torch.autocast('cuda', dtype=self.dtype):
                out = self.model.predict_action(**kw)
        a = out.actions
        a = a.detach().float().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a)
        return np.asarray(a, np.float32).reshape(-1, 7)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--port', type=int, default=7241)
    ap.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16'])
    ap.add_argument('--num-steps', type=int, default=None)
    ap.add_argument('--no-cuda-graph', action='store_true')
    a = ap.parse_args()
    policy = Policy(a.ckpt, a.dtype, a.num_steps, not a.no_cuda_graph)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', a.port))
    srv.listen(4)
    print(f'[molmoact2] listen :{a.port}', flush=True)
    while True:
        (conn, _) = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                req = _recv(conn)
                if req is None:
                    break
                cmd = req.get('cmd')
                if cmd == 'ping':
                    _send(conn, {'ok': True})
                elif cmd == 'infer':
                    t0 = time.time()
                    try:
                        acts = policy.infer(req['images'], req['task'], req['state'])
                        _send(conn, {'actions': acts, 'secs': time.time() - t0})
                    except Exception as e:
                        traceback.print_exc()
                        _send(conn, {'error': f'{type(e).__name__}: {e}'})
                else:
                    _send(conn, {'error': f'unknown cmd {cmd!r}'})
        finally:
            conn.close()
if __name__ == '__main__':
    main()

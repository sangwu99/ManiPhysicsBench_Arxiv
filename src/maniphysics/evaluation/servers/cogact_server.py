from maniphysics.paths import release_root, external_root
import argparse
import os
import socket
import sys
from pathlib import Path
import numpy as np
MPB = release_root()
COGACT = external_root() / 'repos' / 'CogACT'
OPENVLA = Path(os.environ['MPB_OPENVLA_REPO'])
for p in (str(MPB), str(COGACT), str(OPENVLA)):
    sys.path.insert(0, p)
from maniphysics.evaluation.policy_server import _recv, _send

def force_attn_implementation(impl):
    import transformers
    orig = transformers.PreTrainedModel._from_config.__func__

    def patched(cls, config, **kw):
        kw.setdefault('attn_implementation', impl)
        return orig(cls, config, **kw)
    transformers.PreTrainedModel._from_config = classmethod(patched)
    print(f'[cogact] attention override: {impl!r}', flush=True)

def report_attn(policy):
    llm = policy.vla.vlm.llm_backbone.llm
    impl = getattr(llm.config, '_attn_implementation', '?')
    print(f'[cogact] loaded attention implementation = {impl} · llm dtype = {next(llm.parameters()).dtype}', flush=True)
    return impl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--port', type=int, default=7206)
    ap.add_argument('--action-model-type', default='DiT-B')
    ap.add_argument('--use-bf16', action='store_true')
    ap.add_argument('--attn', default=None, choices=['sdpa', 'eager', 'flash_attention_2'])
    a = ap.parse_args()
    if a.attn:
        force_attn_implementation(a.attn)
    print(f"[cogact] loading: {a.ckpt}; gripper {('CONTINUOUS' if os.environ.get('MANIPHYS_CONTINUOUS_GRIPPER') else 'BINARY(stock)')}", flush=True)
    from sim_cogact.cogact_policy import CogACTInference
    policy = CogACTInference(saved_model_path=a.ckpt, policy_setup='widowx_bridge', action_model_type=a.action_model_type, use_bf16=a.use_bf16)
    report_attn(policy)
    policy.reset('warmup')
    policy.step(np.zeros((256, 256, 3), np.uint8), 'warmup')
    policy.reset('warmup')
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', a.port))
    srv.listen(4)
    print(f'[cogact] ready on :{a.port}', flush=True)
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

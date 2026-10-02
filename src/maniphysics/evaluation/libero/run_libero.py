from maniphysics.evaluation.record import finalize_episode
from maniphysics.paths import release_root
import argparse
import json
import os
import socket
import time
from pathlib import Path
import numpy as np
MPB = release_root()
DEFAULT_SUFFIXES = ['Walnut', 'Egg', 'CherryTomatoW30L40', 'Potato', 'Strawberry', 'Plum']

class LiberoClient:

    def __init__(self, port, host='localhost', retries=120):
        from maniphysics.evaluation.policy_server import _recv, _send
        (self._send, self._recv) = (_send, _recv)
        for i in range(retries):
            try:
                self.sock = socket.create_connection((host, port), timeout=600)
                break
            except OSError:
                if i == retries - 1:
                    raise
                time.sleep(5)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def reset(self, instruction):
        self._send(self.sock, dict(cmd='reset', instruction=instruction))
        return self._recv(self.sock)

    def step(self, image, instruction):
        self._send(self.sock, dict(cmd='step', image=np.asarray(image, np.uint8), instruction=instruction))
        r = self._recv(self.sock)
        if r is None or 'error' in r:
            raise RuntimeError(f'Policy server error: {r}')
        return (np.asarray(r['action'], dtype=np.float64), r.get('gripper_raw'))

def get_image(obs, key='agentview_image'):
    return np.ascontiguousarray(np.asarray(obs[key], np.uint8)[::-1, ::-1])

def rollout(env, client, instr, max_steps, chunk_probe=None):
    obs = env.reset()
    client.reset(instr)
    (grips, graws, steps, done) = ([], [], 0, False)
    while steps < max_steps and (not done):
        (act, g_raw) = client.step(get_image(obs), instr)
        grips.append(float(act[6]))
        if g_raw is not None:
            graws.append(float(np.ravel(g_raw)[0]))
        (obs, _, done, _info) = env.step(act)
        steps += 1
        if env.env._check_success():
            done = True
    return (bool(env.env._check_success()), steps, grips, graws)

def ep_row(ep, success, steps, grips, graws, dump_grips=False):
    g = np.asarray(grips) if grips else np.zeros(0)
    g01 = (g + 1.0) / 2.0
    return dict(**{'grips': [round(float(v), 5) for v in g]} if dump_grips else {}, episode=ep, success=success, steps=steps, grip_min=float(g.min()) if g.size else None, grip_max=float(g.max()) if g.size else None, grip_unique=int(len(set(np.round(g, 3)))) if g.size else 0, grip_mid_frac=float(((g01 > 0.05) & (g01 < 0.95)).mean()) if g.size else None, grip_raw_min=float(np.min(graws)) if graws else None, grip_raw_max=float(np.max(graws)) if graws else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=7301)
    ap.add_argument('--envs', nargs='*', default=DEFAULT_SUFFIXES)
    ap.add_argument('--n-ep', type=int, default=12)
    ap.add_argument('--max-steps', type=int, default=600)
    ap.add_argument('--camera-size', type=int, default=256)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dump-grips', action='store_true')
    a = ap.parse_args()
    from maniphysics.bench.libero.tasks import make_maniphys_env
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cont = os.environ.get('MANIPHYS_CONTINUOUS_GRIPPER')
    print(f"[libero] envs={a.envs} n_ep={a.n_ep} max_steps={a.max_steps}; gripper {('CONTINUOUS' if cont else 'BINARY(stock)')}", flush=True)
    client = LiberoClient(a.port)
    summary = {}
    for suffix in a.envs:
        t0 = time.time()
        env = make_maniphys_env(suffix, log_dir=str(out / 'traj' / suffix), camera_size=a.camera_size)
        instr = env.language_instruction
        rows = []
        for ep in range(a.n_ep):
            env.seed(ep)
            (suc, steps, grips, graws) = rollout(env, client, instr, a.max_steps)
            finalize_episode(env, suc, ep, suffix)
            rows.append(ep_row(ep, suc, steps, grips, graws, a.dump_grips))
            print(f"  {suffix} ep{ep}: success={suc} steps={steps} grip∈[{rows[-1]['grip_min']:.2f},{rows[-1]['grip_max']:.2f}] unique {rows[-1]['grip_unique']} intermediate {100 * (rows[-1]['grip_mid_frac'] or 0):.1f}%", flush=True)
        env.close()
        summary[suffix] = dict(env=suffix, instruction=instr, continuous=bool(cont), rows=rows, success=sum((r['success'] for r in rows)), n=len(rows))
        print(f"[{suffix}] successes {summary[suffix]['success']}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
        (out / 'summary_raw.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    tot = sum((v['success'] for v in summary.values()))
    n = sum((v['n'] for v in summary.values()))
    print(f'[libero] total successes {tot}/{n} → {out}')
if __name__ == '__main__':
    main()

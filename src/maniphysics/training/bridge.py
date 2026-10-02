from __future__ import annotations
import numpy as np
BRIDGE_LOWER_M = 0.01
BRIDGE_UPPER_M = 0.035
SIM_LOWER_M = 0.014
SIM_UPPER_M = 0.038
FINGER_OFFSET_M = 0.01945
FINGER_OFFSET_BASE_M = 0.022
FINGER_OFFSET_TIP_M = 0.0179
SIM_GRIPPER_KP = 1000.0
SIM_GRIPPER_FORCE_LIMIT = 60.0

def bridge_g_to_qpos(g):
    return BRIDGE_LOWER_M + (BRIDGE_UPPER_M - BRIDGE_LOWER_M) * np.asarray(g, dtype=np.float64)

def qpos_to_gap(q, offset_m: float=FINGER_OFFSET_M):
    return 2.0 * np.asarray(q, dtype=np.float64) - offset_m

def gap_to_qpos(gap, offset_m: float=FINGER_OFFSET_M):
    return (np.asarray(gap, dtype=np.float64) + offset_m) / 2.0

def qpos_to_sim_g(q):
    return (np.asarray(q, dtype=np.float64) - SIM_LOWER_M) / (SIM_UPPER_M - SIM_LOWER_M)

def qpos_to_bridge_g(q):
    return (np.asarray(q, dtype=np.float64) - BRIDGE_LOWER_M) / (BRIDGE_UPPER_M - BRIDGE_LOWER_M)

def _closed_runs(closed: np.ndarray):
    idx = np.flatnonzero(closed)
    if idx.size == 0:
        return
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    for (i, j) in zip(starts, ends):
        yield (int(i), int(j))

def relabel_episode(state_g: np.ndarray, action_g: np.ndarray, delta_m: float=0.002, mode: str='settle', frame: str='sim', lead: int=1, close_thresh: float=0.5) -> np.ndarray:
    state_g = np.asarray(state_g, dtype=np.float64).reshape(-1)
    action_g = np.asarray(action_g, dtype=np.float64).reshape(-1)
    if state_g.shape != action_g.shape:
        raise ValueError(f'State/action length mismatch: {state_g.shape} vs {action_g.shape}')
    T = state_g.size
    out = action_g.copy()
    if T == 0:
        return out.astype(np.float32)
    closed = action_g < close_thresh
    q_ach = np.full(T, np.nan)
    if mode == 'settle':
        for (i, j) in _closed_runs(closed):
            (lo, hi) = (min(i + lead, T - 1), min(j + lead, T - 1))
            seg = state_g[lo:hi + 1]
            if seg.size == 0:
                seg = state_g[i:j + 1]
            tail = seg[seg.size // 2:]
            q_ach[i:j + 1] = bridge_g_to_qpos(np.median(tail))
    elif mode == 'next':
        nxt = np.minimum(np.arange(T) + lead, T - 1)
        q_ach[closed] = bridge_g_to_qpos(state_g[nxt][closed])
    else:
        raise ValueError(f"Expected mode 'settle' or 'next': {mode}")
    q_tgt = q_ach - float(delta_m)
    g_new = qpos_to_sim_g(q_tgt) if frame == 'sim' else qpos_to_bridge_g(q_tgt)
    m = closed & np.isfinite(g_new)
    out[m] = np.clip(g_new[m], 0.0, 1.0)
    return out.astype(np.float32)

def relabel_dataframe(df, state_key='observation.state', action_key='action', state_dim=7, action_dim=6, **kw):
    s = np.stack(df[state_key].to_numpy())
    a = np.stack(df[action_key].to_numpy())
    a = a.copy()
    a[:, action_dim] = relabel_episode(s[:, state_dim], a[:, action_dim], **kw)
    out = df.copy()
    out[action_key] = list(a.astype(np.float32))
    return out

def relabel_gripper(batch, delta_m: float=0.002, **kw):
    out = dict(batch)
    if 'state.gripper' in batch and 'action.gripper' in batch:
        s = np.asarray(batch['state.gripper'], dtype=np.float64)
        a = np.asarray(batch['action.gripper'], dtype=np.float64)
        if s.shape[-2] != a.shape[-2]:
            raise RuntimeError(f'State/action temporal mismatch ({s.shape[-2]} vs {a.shape[-2]}). Align temporal indices or use the dataframe hook.')
        new = _relabel_stack(s[..., 0], a[..., 0], delta_m=delta_m, **kw)
        out['action.gripper'] = new[..., None].astype(np.float32)
        return out
    if 'state' in batch and 'action' in batch:
        s = np.asarray(batch['state'], dtype=np.float64)
        a = np.asarray(batch['action'], dtype=np.float64)
        if s.shape[-2] != a.shape[-2]:
            raise RuntimeError(f'State/action temporal mismatch ({s.shape[-2]} vs {a.shape[-2]}). Align temporal indices or use the dataframe hook.')
        a = a.copy()
        a[..., 6] = _relabel_stack(s[..., 7], a[..., 6], delta_m=delta_m, **kw)
        out['action'] = a.astype(np.float32)
        return out
    raise KeyError('Missing state/action gripper arrays')

def _relabel_stack(s, a, **kw):
    (s2, a2) = (np.atleast_2d(s), np.atleast_2d(a))
    out = np.stack([relabel_episode(s2[i], a2[i], **kw) for i in range(s2.shape[0])])
    return out.reshape(np.shape(a))

from __future__ import annotations
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

def load_episode_delta_policy(path: str | Path) -> tuple[dict[int, float], dict[str, Any]]:
    policy_path = Path(path).expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f'Missing object-force policy: {policy_path}')
    by_episode: dict[int, float] = {}
    force_counts: Counter[float] = Counter()
    delta_counts: Counter[float] = Counter()
    objects: set[str] = set()
    with policy_path.open(encoding='utf-8') as f:
        for (lineno, line) in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('selection_status', 'selected') != 'selected':
                raise ValueError(f'{policy_path}:{lineno}: expected selected episodes only')
            episode_index = int(row['episode_index'])
            if episode_index in by_episode:
                raise ValueError(f'{policy_path}:{lineno}: episode_index={episode_index} is duplicated')
            delta_mm = float(row['target_delta_mm'])
            force_n = float(row['target_force_N'])
            if not math.isfinite(delta_mm) or delta_mm < 0:
                raise ValueError(f'{policy_path}:{lineno}: invalid target_delta_mm={delta_mm}')
            if not math.isfinite(force_n) or force_n < 0:
                raise ValueError(f'{policy_path}:{lineno}: invalid target_force_N={force_n}')
            gain = float(row['kp_N_m'])
            if not math.isfinite(gain) or gain <= 0 or not math.isclose(delta_mm, 1000.0 * force_n / gain, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f'{policy_path}:{lineno}: force and offset do not follow the nominal servo relation')
            if row['force_convention'] != 'per_finger':
                raise ValueError(f'{policy_path}:{lineno}: expected per-finger forces')
            by_episode[episode_index] = delta_mm / 1000.0
            force_counts[force_n] += 1
            delta_counts[delta_mm] += 1
            if row.get('clean_object'):
                objects.add(str(row['clean_object']))
    if not by_episode:
        raise ValueError(f'Empty object-force policy: {policy_path}')
    metadata = {'path': str(policy_path), 'episode_count': len(by_episode), 'object_count': len(objects), 'force_counts_N': dict(sorted(force_counts.items())), 'delta_counts_mm': dict(sorted(delta_counts.items()))}
    return (by_episode, metadata)

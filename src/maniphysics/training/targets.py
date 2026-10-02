import argparse
import hashlib
import json
import math
from pathlib import Path

from maniphysics.zoo.catalog import object_id


def target(entry, kp_N_m=1000.0, cap_N=4.0):
    lift = float(entry['lift_force_N'])
    if not math.isfinite(lift) or lift < 0 or not math.isfinite(kp_N_m) or kp_N_m <= 0 or not math.isfinite(cap_N) or cap_N <= 0:
        raise ValueError('Invalid lifting force, servo gain, or force cap')
    if entry.get('zoo_object_id'):
        object_id(entry['zoo_object_id'])
    if 'damage_force_N' in entry:
        damage = float(entry['damage_force_N'])
        if not math.isfinite(damage) or damage <= lift:
            raise ValueError('The damage threshold must exceed the lifting requirement')
        force = (lift + damage) / 2
        method = 'safe_interval_midpoint'
    else:
        margin = float(entry['grip_margin_N'])
        if not math.isfinite(margin) or margin <= 0:
            raise ValueError('A positive grip margin is required for a lifting proxy')
        force = lift + margin
        method = 'lifting_proxy_plus_margin'
    force = min(force, cap_N)
    return dict(target_force_N=force, target_delta_mm=1000 * force / kp_N_m, method=method, kp_N_m=kp_N_m, force_convention='per_finger', evidence=entry)


def build(selected, categories, kp_N_m=1000.0, cap_N=4.0):
    policy = []
    seen = set()
    targets = {key: target(value, kp_N_m, cap_N) for key, value in categories.items()}
    for row in selected:
        episode = row['episode_index']
        if type(episode) is not int or episode < 0 or episode in seen:
            raise ValueError('Episode indices must be unique nonnegative integers')
        seen.add(episode)
        policy.append(dict(episode_index=episode, clean_object=row['clean_object'], selection_status='selected', **targets[row['clean_object']]))
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected', type=Path, required=True)
    parser.add_argument('--categories', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kp', type=float, default=1000.0)
    parser.add_argument('--cap', type=float, default=4.0)
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.selected.read_text().splitlines() if s.strip()]
    categories = json.loads(args.categories.read_text())
    policy = build(rows, categories, args.kp, args.cap)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'candidate_episodes.jsonl'
    path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in policy))
    summary = dict(episodes=len(policy), objects=len({x['clean_object'] for x in policy}), policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), force_to_offset=dict(version='nominal_per_finger_kp', kp_N_m=args.kp), force_cap_N=args.cap)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()

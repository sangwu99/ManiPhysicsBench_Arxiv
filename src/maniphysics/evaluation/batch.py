import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from maniphysics.assessment.cache import save_json, verify
from maniphysics.assessment.grasp import extract
from maniphysics.assessment.pipeline import assess, aggregate
from maniphysics.assessment.runtime import Runtime


def collect_episodes(roots):
    episodes = []
    seen = set()
    for root in roots:
        for path in sorted(Path(root).rglob('episodes.jsonl')):
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for row in rows:
                trajectory = (path.parent / row['trajectory']).resolve()
                if trajectory in seen:
                    raise ValueError(f'Duplicate trajectory: {trajectory}')
                seen.add(trajectory)
                if row.get('error'):
                    raise ValueError(f'Rollout execution failed: {trajectory}')
                meta = json.loads(trajectory.with_suffix('.json').read_text())
                if meta['task'] != row['task'] or meta['evaluator_episode'] != row['episode'] or meta['success'] != row['task_success']:
                    raise ValueError(f'Episode manifest mismatch: {trajectory}')
                data = extract(trajectory, meta['model_id'])
                data.update(trajectory=str(trajectory), task=row['task'], evaluator_episode=row['episode'])
                episodes.append(data)
    if not episodes:
        raise ValueError('No recorded episode manifests found')
    return episodes


def collect_donors(cache_roots):
    donors = {}
    for root in cache_roots:
        for path in Path(root).rglob('verdict_*.json'):
            item = json.loads(path.read_text())
            if item['origin'] != 'direct' or item['threshold']['kind'] != 'bracketed':
                continue
            verify(path.parent)
            donors[item['compatibility'], item['id']] = item
    return list(donors.values())


def direct_episode(index, episode, cache, runtime, coupled, stop_after_damage):
    results = []
    for grasp in episode['grasps']:
        result = assess(grasp, cache, Runtime(**runtime), coupled=coupled)
        results.append(result)
        if stop_after_damage and result['verdict']['status'] == 'damaged':
            break
    return index, results


def run(episodes, output, runtime, *, workers=1, donor_roots=(), coupled=True, stop_after_damage=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'cache'
    results = {}
    errors = {}
    save_json(output / 'inputs.json', episodes)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(direct_episode, i, e, cache, runtime, coupled, stop_after_damage): i for i, e in enumerate(episodes)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                _, values = future.result()
                results[i] = values
                save_json(output / f'direct_{i:05d}.json', values)
            except Exception as error:
                errors[i] = dict(type=type(error).__name__, message=str(error))
                save_json(output / f'error_{i:05d}.json', errors[i])
    donors = collect_donors([cache, *donor_roots])
    save_json(output / 'direct_donors.json', donors)
    final = []
    for i, episode in enumerate(episodes):
        if i in errors:
            final.append(dict(episode=i, state='error', error=errors[i]))
            continue
        values = results[i]
        for j, value in enumerate(values):
            if value['verdict']['status'] == 'unknown':
                values[j] = assess(episode['grasps'][j], cache, Runtime(**runtime), coupled=coupled, donors=donors)
        result = aggregate(values, task_success=episode['task_success'], coverage_complete=not episode['coverage_reasons'] and len(values) == len(episode['grasps']))
        result.update(episode=i, task=episode['task'], evaluator_episode=episode['evaluator_episode'], trajectory=episode['trajectory'], grasp_results=values, input_provenance=episode['provenance'], coverage_reasons=episode['coverage_reasons'], unassessed_grasps=len(episode['grasps']) - len(values))
        save_json(output / f'episode_{i:05d}.json', result)
        final.append(result)
    save_json(output / 'summary.json', dict(episodes=final, errors=len(errors), direct_donors=len(donors)))
    if errors:
        raise RuntimeError(f'{len(errors)} episodes could not be assessed; see saved error records')
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--donor-roots', nargs='*', type=Path, default=[])
    parser.add_argument('--raw-force', action='store_true')
    parser.add_argument('--stop-after-damage', action='store_true')
    args = parser.parse_args()
    episodes = collect_episodes(args.roots)
    run(episodes, args.output, json.loads(args.runtime.read_text()), workers=args.workers, donor_roots=args.donor_roots, coupled=not args.raw_force, stop_after_damage=args.stop_after_damage)


if __name__ == '__main__':
    main()

import argparse
import json
from pathlib import Path
from .runtime import Runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['assess', 'prepare', 'extract', 'episode'])
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--raw-force', action='store_true')
    parser.add_argument('--donors', type=Path)
    parser.add_argument('--object')
    parser.add_argument('--kp', type=float)
    parser.add_argument('--actuator-limit', type=float)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--task')
    parser.add_argument('--summary-episode', type=int)
    args = parser.parse_args()
    if args.command == 'extract':
        from .grasp import extract
        if args.object is None:
            parser.error('--object is required for extraction')
        data = extract(args.input, args.object, kp_N_m=args.kp, actuator_limit_N=args.actuator_limit,
                       summary=args.summary, task=args.task, summary_episode=args.summary_episode)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2, allow_nan=False))
        return
    if args.runtime is None:
        parser.error('--runtime is required for FEM execution')
    from .pipeline import assess
    runtime = Runtime(**json.loads(args.runtime.read_text()))
    donors = json.loads(args.donors.read_text()) if args.donors else []
    if args.command == 'episode':
        from .pipeline import aggregate
        from .cache import save_json
        episode = json.loads(args.input.read_text())
        if not isinstance(episode.get('task_success'), bool):
            parser.error('Episode input must contain a boolean task_success')
        results = [assess(g, args.output / 'cache', runtime,
                          coupled=not args.raw_force, donors=donors) for g in episode['grasps']]
        result = aggregate(results, task_success=episode['task_success'],
                           coverage_complete=not episode['coverage_reasons'])
        result['grasp_results'] = results
        result['input_provenance'] = episode.get('provenance')
        args.output.mkdir(parents=True, exist_ok=True)
        save_json(args.output / 'assessment.json', result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    result = assess(json.loads(args.input.read_text()), args.output, runtime,
                    coupled=not args.raw_force, donors=donors, prepare_only=args.command == 'prepare')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

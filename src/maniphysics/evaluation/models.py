import argparse
import json
import os
from pathlib import Path

from maniphysics.paths import release_root
from maniphysics.evaluation.launch import run


def configuration(model, runtime, output, tasks, episodes, gpu, port):
    root = release_root()
    catalog = json.loads((root / 'bench/evaluation/models.json').read_text())
    entry = catalog[model]
    context = dict(root=str(root), external=runtime['external_root'], episodes=str(episodes), gpu=str(gpu), port=str(port), **runtime.get('variables', {}))
    context['checkpoint'] = runtime.get('checkpoints', {}).get(model, entry['checkpoint'].format_map(context))
    base_env = dict(CUDA_VISIBLE_DEVICES=str(gpu), MANIPHYS_CONTINUOUS_GRIPPER='1', MANIPHYS_DISTRACTORS='0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', XLA_PYTHON_CLIENT_PREALLOCATE='false', MPB_EXTERNAL_ROOT=runtime['external_root'], MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=str(gpu), VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')
    base_env.update(runtime.get('env', {}))

    def expand(spec, values):
        env = dict(base_env)
        env.update({k: v.format_map(values) for k, v in spec['env'].items()})
        env['PYTHONPATH'] = os.pathsep.join(x.format_map(values) for x in spec['pythonpath'])
        command = [runtime['python'][spec['python']], *[s.format_map(values) for s in spec['arguments']]]
        if runtime.get('cpu_affinity'):
            command = ['taskset', '-c', ','.join(map(str, runtime['cpu_affinity'])), *command]
        return dict(command=command, cwd=spec['cwd'].format_map(values), env=env)

    config = dict(model=model, jobs=[])
    if 'server' in entry:
        config['server'] = expand(entry['server'], context)
        config['server']['ready'] = dict(host='127.0.0.1', port=port, timeout_s=runtime.get('server_timeout_s', 5400))
    for task in tasks:
        values = dict(context, task=task, rollout=str(output / 'rollouts' / task))
        job = expand(entry['runner'], values)
        job['name'] = task
        job['env']['MPB_RUNTIME_ROOT'] = str(output / 'generated' / task)
        config['jobs'].append(job)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--port', type=int, default=7820)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = configuration(args.model, json.loads(args.runtime.read_text()), output, args.tasks, args.episodes, args.gpu, args.port)
    path = output / 'launch.json'
    path.write_text(json.dumps(config, indent=2) + '\n')
    if not args.prepare_only:
        run(path, output / 'launcher')


if __name__ == '__main__':
    main()

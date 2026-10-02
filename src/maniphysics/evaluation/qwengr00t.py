import argparse
import json
from pathlib import Path
from .launch import run

def configuration(args):
    output = args.output.resolve()
    starvla = args.starvla_root.resolve()
    bench = args.bench_root.resolve()
    simpler = args.simpler_root.resolve()
    checkpoint = args.checkpoint.resolve()
    server_script = starvla / 'deployment/model_server/server_policy.py'
    for path in (args.server_python, args.sim_python, args.runner, checkpoint, server_script):
        if not path.is_file():
            raise FileNotFoundError(path)
    server = dict(command=[str(args.server_python.resolve()), str(server_script), '--ckpt_path', str(checkpoint), '--port', str(args.port), '--use_bf16', '--idle_timeout', '-1'], cwd=str(starvla), env={'CUDA_VISIBLE_DEVICES': str(args.gpu)}, ready=dict(host='127.0.0.1', port=args.port, timeout_s=600))
    jobs = []
    for (index, env) in enumerate(args.envs):
        directory = output / f'rollout_{index:02d}'
        (directory / 'traj').mkdir(parents=True)
        jobs.append(dict(name=env, cwd=str(bench), command=[str(args.sim_python.resolve()), str(args.runner.resolve()), '--env', env, '--episodes', str(args.episodes), '--port', str(args.port), '--out-dir', str(directory), '--client', 'starvla', '--ckpt', str(checkpoint)], env={'CUDA_VISIBLE_DEVICES': str(args.gpu), 'VK_ICD_FILENAMES': args.vulkan_icd, 'PYTHONPATH': ':'.join(map(str, (bench, simpler, starvla))), 'MANIPHYS_CONTINUOUS_GRIPPER': '1', 'MANIPHYS_LOG_DIR': str(directory / 'traj')}))
    return dict(server=server, jobs=jobs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--port', type=int)
    for option in ('checkpoint', 'server-python', 'sim-python', 'starvla-root', 'simpler-root', 'bench-root', 'runner', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--envs', nargs='+', required=True)
    parser.add_argument('--vulkan-icd', default='/etc/vulkan/icd.d/nvidia_icd.json')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.port is None:
        args.port = 6940 + args.gpu
    if args.output.exists():
        parser.error('Output directory already exists')
    config = configuration(args)
    path = args.output / 'launch.json'
    path.write_text(json.dumps(config, indent=2))
    if not args.prepare_only:
        run(path, args.output / 'logs')
if __name__ == '__main__':
    main()

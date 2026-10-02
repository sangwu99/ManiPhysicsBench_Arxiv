import argparse
import hashlib
import json
import math
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--base-vlm', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--deepspeed-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-output', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--per-gpu', type=int, default=12)
    parser.add_argument('--accumulation', type=int, default=3)
    parser.add_argument('--sample-budget', type=int, default=5120000)
    args = parser.parse_args()
    config = yaml.safe_load(args.source_config.read_text())
    config['framework']['qwenvl']['base_vlm'] = str(args.base_vlm.resolve())
    config['framework']['action_model']['repeated_diffusion_steps'] = config['trainer']['repeated_diffusion_steps']
    config['datasets']['vla_data'].update(data_root_dir=str(args.dataset_root.resolve()), data_mix='bridge', video_backend='torchvision_av', per_device_batch_size=args.per_gpu, sequential_step_sampling=False)
    config['trainer']['gradient_accumulation_steps'] = args.accumulation
    config['trainer']['freeze_modules'] = ''
    if config['trainer'].get('is_resume') or config['trainer'].get('pretrained_checkpoint'):
        raise ValueError('Expected base initialization without a resumed checkpoint')
    config.update(run_root_dir=str(args.training_output.resolve().parent), run_id=args.training_output.name, trackers=['jsonl'], is_debug=False)
    count = len(args.gpus.split(','))
    batch = count * args.per_gpu * args.accumulation
    steps = math.ceil(args.sample_budget / batch)
    if steps > config['trainer']['max_train_steps']:
        raise ValueError('Sample budget exceeds the inherited schedule')
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = args.output / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    ds = json.loads(args.deepspeed_config.read_text())
    ds['gradient_accumulation_steps'] = args.accumulation
    ds_path = args.output / 'deepspeed.json'
    ds_path.write_text(json.dumps(ds, indent=2))
    ac = dict(compute_environment='LOCAL_MACHINE', debug=False, distributed_type='DEEPSPEED', num_machines=1, num_processes=count, deepspeed_config=dict(deepspeed_config_file=str(ds_path.resolve()), zero3_init_flag=False, deepspeed_multinode_launcher='standard'))
    (args.output / 'accelerate.yaml').write_text(yaml.safe_dump(ac))
    plan = dict(gpus=args.gpus, per_gpu=args.per_gpu, accum=args.accumulation, global_batch=batch, sample_budget=args.sample_budget, stop_after_steps=steps, scheduler_horizon=config['trainer']['max_train_steps'], policy=str(args.policy.resolve()), policy_sha256=file_hash(args.policy), config_sha256=file_hash(config_path), force_to_offset=dict(version='nominal_per_finger_kp'))
    (args.output / 'manifest.json').write_text(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()

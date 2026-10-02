import argparse
import json
import os
from pathlib import Path
import subprocess

from maniphysics.paths import release_root
from maniphysics.training.prepare import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--accelerate', required=True)
    parser.add_argument('--external-root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=29640)
    args = parser.parse_args()
    prepared = args.prepared.resolve()
    manifest = json.loads((prepared / 'manifest.json').read_text())
    if file_hash(manifest['policy']) != manifest['policy_sha256'] or file_hash(prepared / 'config.yaml') != manifest['config_sha256']:
        raise ValueError('Prepared inputs have changed')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=manifest['gpus'], ACCELERATE_GRADIENT_ACCUMULATION_STEPS=str(manifest['accum']), MPB_EXTERNAL_ROOT=str(args.external_root.resolve()), MANIPHYS_RELABEL='1', MANIPHYS_RELABEL_POLICY_JSONL=manifest['policy'], MANIPHYS_LORA_R='32', MANIPHYS_LORA_ALPHA='64', MANIPHYS_LORA_DROPOUT='0.05', MANIPHYS_LORA_TARGETS='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj', MANIPHYS_HEAD_ONLY='0', TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled')
    environment['PYTHONPATH'] = os.pathsep.join([str(release_root() / 'src'), str(args.external_root.resolve() / 'repos/starVLA')])
    command = [args.accelerate, 'launch', '--config_file', str(prepared / 'accelerate.yaml'), '--num_processes', str(len(manifest['gpus'].split(','))), '--main_process_port', str(args.port), '-m', 'maniphysics.training.qwen_train', '--config', str(prepared / 'config.yaml'), '--manifest', str(prepared / 'manifest.json'), '--stop-after-steps', str(manifest['stop_after_steps'])]
    subprocess.run(command, env=environment, check=True)


if __name__ == '__main__':
    main()

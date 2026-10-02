from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import time

def validate_trainable(model):
    counts = {'lora': 0, 'action_model': 0, 'frozen': 0}
    for (name, p) in model.named_parameters():
        kind = 'action_model' if name.startswith('action_model.') else 'lora' if '.lora_' in name else 'frozen'
        if p.requires_grad != (kind != 'frozen'):
            raise RuntimeError(f'Unexpected trainability: {name}, requires_grad={p.requires_grad}')
        counts[kind] += p.numel()
    if not counts['action_model'] or not counts['lora']:
        raise RuntimeError(f'Missing action head/LoRA: {counts}')
    return counts

def main(args):
    import maniphysics.training.qwen_base as base
    from accelerate.utils import set_seed
    from omegaconf import OmegaConf
    from tqdm import tqdm
    T = base.T
    cfg = OmegaConf.load(args.config)
    from maniphysics.training.data_compat import install_release_packing
    install_release_packing(cfg.datasets.vla_data.image_size)
    plan = json.loads(args.manifest.read_text())
    if not 0 < args.stop_after_steps <= cfg.trainer.max_train_steps:
        raise ValueError('Stop budget must be within the inherited schedule')
    if T.accelerator.gradient_accumulation_steps != cfg.trainer.gradient_accumulation_steps:
        raise RuntimeError('Accelerate and trainer accumulation differ')
    if T.accelerator.state.deepspeed_plugin.deepspeed_config['gradient_accumulation_steps'] != cfg.trainer.gradient_accumulation_steps:
        raise RuntimeError('DeepSpeed and trainer accumulation differ')
    if T.accelerator.num_processes * cfg.datasets.vla_data.per_device_batch_size * cfg.trainer.gradient_accumulation_steps != plan['global_batch']:
        raise RuntimeError('Actual global batch differs from prepared plan')
    T.accelerator.gradient_state.plugin_kwargs['sync_each_batch'] = True
    T.accelerator.gradient_state.plugin_kwargs['sync_with_dataloader'] = False
    set_seed(int(cfg.seed))
    out = Path(cfg.run_root_dir) / cfg.run_id
    if (out / 'trainability.json').exists():
        raise FileExistsError(f'Run already initialized; do not overwrite: {out}')
    if T.accelerator.is_main_process:
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.manifest, out / 'input_manifest.json')
        shutil.copy2(Path(os.environ['MANIPHYS_RELABEL_POLICY_JSONL']), out / 'candidate_episodes.jsonl')
        print(f"[nominal] policy_sha256={plan['policy_sha256']} version={plan['force_to_offset']['version']}", flush=True)
    T.accelerator.wait_for_everyone()
    original_optimizer = base.build_optimizer

    def checked_optimizer(model, config):
        counts = validate_trainable(model)
        (optimizer, scheduler) = original_optimizer(model, config)
        actual = [id(p) for g in optimizer.param_groups for p in g['params']]
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise RuntimeError('Optimizer does not cover each trainable parameter exactly once')
        if T.accelerator.is_main_process:
            (out / 'trainability.json').write_text(json.dumps(counts, indent=2))
        return (optimizer, scheduler)
    base.build_optimizer = checked_optimizer

    class BudgetTrainer(T.VLATrainer):

        def train(self):
            self._log_training_config()
            self._save_config_snapshot()
            self._create_data_iterators()
            progress = tqdm(total=args.stop_after_steps, disable=not self.accelerator.is_local_main_process)
            microsteps = 0
            while self.completed_steps < args.stop_after_steps:
                t0 = time.perf_counter()
                batch = self._get_next_batch()
                t1 = time.perf_counter()
                metrics = self._train_step(batch)
                t2 = time.perf_counter()
                microsteps += 1
                if not self.accelerator.sync_gradients:
                    continue
                self.completed_steps += 1
                progress.update(1)
                progress.set_postfix(loss=metrics['action_dit_loss'], model_s=round(t2 - t1, 3))
                if self.completed_steps % self.config.trainer.eval_interval == 0:
                    metrics = self.eval_action_model(metrics)
                metrics.update(data_time=t1 - t0, model_time=t2 - t1)
                self._log_metrics(metrics)
                if self.completed_steps % self.config.trainer.save_interval == 0:
                    self._save_checkpoint()
                    self.accelerator.wait_for_everyone()
            if self.completed_steps % self.config.trainer.save_interval:
                self._save_checkpoint()
                self.accelerator.wait_for_everyone()
            self._finalize_training()
            if self.accelerator.is_main_process:
                (out / 'budget_completed.json').write_text(json.dumps({'optimizer_steps': self.completed_steps, 'microsteps': microsteps, 'training_samples': microsteps * plan['per_gpu'] * self.accelerator.num_processes, 'scheduler_horizon': self.config.trainer.max_train_steps}, indent=2))
            progress.close()
    T.VLATrainer = BudgetTrainer
    base.main(cfg, ckpt_path=None, initial_step=0)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--stop-after-steps', type=int, required=True)
    main(p.parse_args())

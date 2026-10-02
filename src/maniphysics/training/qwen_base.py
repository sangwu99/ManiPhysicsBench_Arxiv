from __future__ import annotations
from maniphysics.paths import external_root
import os
import sys
from pathlib import Path
STARVLA = external_root() / 'repos' / 'starVLA'
for p in (str(STARVLA),):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(STARVLA)
import torch
from maniphysics.training.qwen_patch import inject_lora, merge_lora_state_dict, patch_relabel
from starVLA.model.framework.base_framework import build_framework
from starVLA.training import train_starvla as T
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
from transformers import get_scheduler

def _shim_save_full_config():
    from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig
    orig = AccessTrackedConfig.save_full_config
    if getattr(orig, '_maniphys_shimmed', False):
        return

    def save_full_config(self, filepath, **_ignored):
        return orig(self, filepath)
    save_full_config._maniphys_shimmed = True
    AccessTrackedConfig.save_full_config = save_full_config

def load_released_checkpoint(model, ckpt_path: Path):
    sd = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
    (missing, unexpected) = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}')
    print(f'[maniphys] released ckpt loaded strictly: {ckpt_path} ({len(sd)} tensors)')
    del sd
    return model

def build_optimizer(model, cfg):
    groups = build_param_lr_groups(model=model, cfg=cfg)
    kept = []
    for g in groups:
        params = [p for p in g['params'] if p.requires_grad]
        if params:
            kept.append({**g, 'params': params})
            print(f"[maniphys] LR group {g['name']}: lr={g['lr']} n={len(params)}")
    optimizer = torch.optim.AdamW(kept, lr=cfg.trainer.learning_rate.base, betas=tuple(cfg.trainer.optimizer.betas), weight_decay=cfg.trainer.optimizer.weight_decay, eps=cfg.trainer.optimizer.eps)
    lr_scheduler = get_scheduler(name=cfg.trainer.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=cfg.trainer.num_warmup_steps, num_training_steps=cfg.trainer.max_train_steps, scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs)
    return (optimizer, lr_scheduler)

def _patch_merged_checkpoint_save() -> None:
    if getattr(T.VLATrainer, '_maniphys_merged_save', False):
        return

    def wrap(original):

        def patched(self, *args, **kwargs):
            original_get_state_dict = self.accelerator.get_state_dict

            def get_merged_state_dict(model):
                state_dict = original_get_state_dict(model)
                (merged, n_merged) = merge_lora_state_dict(state_dict)
                if n_merged <= 0:
                    raise RuntimeError('No LoRA layers to merge')
                self.accelerator.print(f'[training] Merged LoRA layers: {n_merged} merged')
                return merged
            self.accelerator.get_state_dict = get_merged_state_dict
            try:
                return original(self, *args, **kwargs)
            finally:
                self.accelerator.get_state_dict = original_get_state_dict
        return patched
    T.VLATrainer._save_checkpoint = wrap(T.VLATrainer._save_checkpoint)
    T.VLATrainer._finalize_training = wrap(T.VLATrainer._finalize_training)
    T.VLATrainer._maniphys_merged_save = True

def main(cfg, ckpt_path: Path | None, initial_step: int=0):
    patch_relabel()
    _shim_save_full_config()
    cfg = T.wrap_config(cfg)
    output_dir = T.setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    if ckpt_path is None:
        print('[training] Initialization: base VLM and new action model')
    else:
        load_released_checkpoint(vla, ckpt_path)
    if os.environ.get('MANIPHYS_HEAD_ONLY', '0') == '1':
        for (n, p) in vla.named_parameters():
            p.requires_grad = n.startswith('action_model.')
        n_head = sum((p.numel() for p in vla.parameters() if p.requires_grad))
        print(f'[maniphys] HEAD_ONLY: action_model {n_head / 1000000.0:.1f}M trainable; no LoRA', flush=True)
    else:
        vla = inject_lora(vla)
        _patch_merged_checkpoint_save()
    vla_train_dataloader = T.prepare_data(cfg=cfg, accelerator=T.accelerator, output_dir=output_dir)
    (optimizer, lr_scheduler) = build_optimizer(vla, cfg)
    trainer = T.VLATrainer(cfg=cfg, model=vla, vla_train_dataloader=vla_train_dataloader, optimizer=optimizer, lr_scheduler=lr_scheduler, accelerator=T.accelerator)
    trainer.prepare_training()
    trainer.train()
    print(f'[maniphys] peak GPU alloc={torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB  reserved={torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GiB')
    import torch.distributed as dist
    dist.barrier()
    dist.destroy_process_group()

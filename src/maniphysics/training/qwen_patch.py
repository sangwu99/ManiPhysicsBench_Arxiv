from __future__ import annotations
from maniphysics.paths import release_root, external_root
import os
import sys
MPB = release_root()
STARVLA = external_root() / 'repos' / 'starVLA'
for p in (str(STARVLA), str(MPB)):
    if p not in sys.path:
        sys.path.insert(0, p)
RELABEL_DATASETS = {'bridge_orig_1.0.0_lerobot', 'bridge_orig_lerobot'}
_ATTR_FLAG = '_maniphys_relabeled'
POLICY_ENV = 'MANIPHYS_RELABEL_POLICY_JSONL'

def relabel_enabled() -> bool:
    return os.environ.get('MANIPHYS_RELABEL', '') not in ('', '0', 'false', 'False')

def relabel_delta_m() -> float:
    return float(os.environ.get('MANIPHYS_RELABEL_DELTA', '0.002'))

def patch_relabel() -> bool:
    if not relabel_enabled():
        print('[training] Relabeling disabled; original labels')
        return False
    from maniphysics.training.bridge import relabel_dataframe
    from maniphysics.training.object_force_policy import load_episode_delta_policy
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
    from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
    delta_m = relabel_delta_m()
    policy_path = os.environ[POLICY_ENV]
    (episode_delta, policy_meta) = load_episode_delta_policy(policy_path)
    _orig_get_traj = LeRobotSingleDataset.get_trajectory_data

    def _get_trajectory_data(self, trajectory_id):
        df = _orig_get_traj(self, trajectory_id)
        if df is None or self.dataset_name not in RELABEL_DATASETS:
            return df
        episode_index = int(trajectory_id)
        if policy_meta is not None and episode_index not in episode_delta:
            return df
        if df.attrs.get(_ATTR_FLAG):
            return df
        applied_delta_m = episode_delta.get(episode_index, delta_m)
        out = relabel_dataframe(df, delta_m=applied_delta_m)
        out.attrs[_ATTR_FLAG] = True
        out.attrs['_maniphys_delta_m'] = applied_delta_m
        return out
    LeRobotSingleDataset.get_trajectory_data = _get_trajectory_data
    bridge_cfg = ROBOT_TYPE_CONFIG_MAP['oxe_bridge']
    _orig_transform = bridge_cfg.transform

    def _transform():
        composed = _orig_transform()
        n_dropped = 0
        for t in composed.transforms:
            modes = getattr(t, 'normalization_modes', None)
            if modes and 'action.gripper' in modes:
                modes.pop('action.gripper')
                n_dropped += 1
        assert n_dropped == 1, f'action.gripper normalization sites: {n_dropped} (expected one)'
        return composed
    bridge_cfg.transform = _transform
    if policy_meta is None:
        source = f'global delta_m={delta_m}'
    else:
        source = f"policy={policy_meta['path']} ({policy_meta['episode_count']} episodes/{policy_meta['object_count']} objects)"
    print(f'[maniphys] RELABEL ON — {source}, mode=settle frame=sim')
    return True
LORA_R = int(os.environ.get('MANIPHYS_LORA_R', '32'))
LORA_ALPHA = int(os.environ.get('MANIPHYS_LORA_ALPHA', '64'))
LORA_DROPOUT = float(os.environ.get('MANIPHYS_LORA_DROPOUT', '0.05'))
LORA_TARGETS = [x.strip() for x in os.environ.get('MANIPHYS_LORA_TARGETS', 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj').split(',') if x.strip()]
FULL_TRAIN_PREFIXES = ('action_model.',)

def inject_lora(model):
    from peft import LoraConfig, inject_adapter_in_model
    lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias='none', target_modules=LORA_TARGETS)
    if not hasattr(model, 'qwen_vl_interface') or not hasattr(model.qwen_vl_interface, 'model'):
        raise AttributeError('Missing qwen_vl_interface.model')
    inject_adapter_in_model(lora_cfg, model.qwen_vl_interface.model)
    n_lora = n_full = 0
    for (name, p) in model.named_parameters():
        is_lora = 'lora_' in name
        is_full = name.startswith(FULL_TRAIN_PREFIXES)
        p.requires_grad = bool(is_lora or is_full)
        if is_lora:
            n_lora += p.numel()
        elif is_full:
            n_full += p.numel()
    total = sum((p.numel() for p in model.parameters()))
    print(f'[maniphys] LoRA r={LORA_R} a={LORA_ALPHA} drop={LORA_DROPOUT} targets={LORA_TARGETS}')
    print(f'[maniphys] trainable: lora {n_lora / 1000000.0:.3f}M + full {n_full / 1000000.0:.3f}M = {(n_lora + n_full) / 1000000.0:.3f}M / {total / 1000000.0:.3f}M ({100 * (n_lora + n_full) / total:.3f}%)')
    assert n_lora > 0 and n_full > 0, 'Missing LoRA or full-training parameters'
    return model

def merge_lora_state_dict(state_dict):
    import torch
    scaling = LORA_ALPHA / LORA_R
    (merged, n_merged) = ({}, 0)
    lora_A = {k[:-len('.lora_A.default.weight')]: v for (k, v) in state_dict.items() if k.endswith('.lora_A.default.weight')}
    lora_B = {k[:-len('.lora_B.default.weight')]: v for (k, v) in state_dict.items() if k.endswith('.lora_B.default.weight')}
    assert set(lora_A) == set(lora_B), 'LoRA A/B keys do not match'
    for (k, v) in state_dict.items():
        if '.lora_' in k:
            continue
        if '.base_layer.' in k:
            (mod, tail) = k.split('.base_layer.', 1)
            new_k = f'{mod}.{tail}'
            if tail == 'weight' and mod in lora_A:
                A = lora_A[mod].to(torch.float32)
                B = lora_B[mod].to(torch.float32)
                v = (v.to(torch.float32) + scaling * (B @ A)).to(v.dtype)
                n_merged += 1
            merged[new_k] = v
        else:
            merged[k] = v
    assert n_merged == len(lora_A), f'Incomplete merge: {n_merged} != {len(lora_A)}'
    return (merged, n_merged)

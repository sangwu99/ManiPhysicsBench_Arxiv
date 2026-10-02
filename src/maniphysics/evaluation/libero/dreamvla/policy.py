from maniphysics.paths import release_root, external_root
import sys
from collections import deque
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
MPB = release_root()
DREAMVLA_ROOT = external_root() / 'repos' / 'DreamVLA'
EVAL_ARGV = ['--traj_cons', '--rgb_pad', '10', '--gripper_pad', '4', '--gradient_accumulation_steps', '1', '--bf16_module', 'vision_encoder', '--calvin_dataset', '', '--libero_path', 'LIBERO', '--workers', '16', '--lr_scheduler', 'cosine', '--save_every_iter', '50000', '--num_epochs', '20', '--seed', '66', '--batch_size', '64', '--precision', 'fp32', '--weight_decay', '1e-4', '--num_resampler_query', '16', '--run_name', 'test', '--transformer_layers', '24', '--hidden_dim', '1024', '--transformer_heads', '16', '--phase', 'evaluate', '--finetune_type', 'libero_object', '--action_pred_steps', '3', '--future_steps', '3', '--sequence_length', '7', '--obs_pred', '--gripper_width', '--eval_libero_ensembling', '--use_dit_head', '--load_track_labels', '--load_sam_features', '--sam_feat_pred', '--loss_sam_feat', '--flow_as_mask', '--attn_implementation', 'sdpa', '--save_checkpoint_path', '/tmp/dreamvla_unused']

def _install_path():
    if str(DREAMVLA_ROOT) not in sys.path:
        sys.path.insert(0, str(DREAMVLA_ROOT))

def build_args(vit_checkpoint_path):
    _install_path()
    from utils.arguments_utils import get_parser
    argv = list(EVAL_ARGV) + ['--vit_checkpoint_path', str(vit_checkpoint_path)]
    saved = sys.argv
    try:
        sys.argv = ['eval_libero.py'] + argv
        args = get_parser(is_eval=True).parse_args(argv)
    finally:
        sys.argv = saved
    (args.local_rank, args.rank, args.world_size) = (0, 0, 1)
    return args

def build_model(ckpt_path, vit_checkpoint_path, device='cuda'):
    _install_path()
    from models.dreamvla_model import DreamVLA
    args = build_args(vit_checkpoint_path)
    model = DreamVLA(finetune_type=args.finetune_type, clip_device=device, vit_checkpoint_path=args.vit_checkpoint_path, sequence_length=args.sequence_length, num_resampler_query=args.num_resampler_query, num_obs_token_per_image=args.num_obs_token_per_image, calvin_input_image_size=args.calvin_input_image_size, patch_size=args.patch_size, action_pred_steps=args.action_pred_steps, obs_pred=args.obs_pred, atten_only_obs=args.atten_only_obs, attn_robot_proprio_state=args.attn_robot_proprio_state, atten_goal=args.atten_goal, atten_goal_state=args.atten_goal_state, mask_l_obs_ratio=args.mask_l_obs_ratio, transformer_layers=args.transformer_layers, hidden_dim=args.hidden_dim, transformer_heads=args.transformer_heads, phase=args.phase, gripper_width=args.gripper_width, depth_pred=args.depth_pred, trajectory_pred=args.trajectory_pred, pred_num=args.pred_num, use_trajectory_query=args.use_trajectory_query, track_label_patch_size=args.track_label_patch_size, use_dinosiglip=args.use_dinosiglip, use_dit_head=args.use_dit_head, dino_feat_pred=args.dino_feat_pred, sam_feat_pred=args.sam_feat_pred, no_pred_gripper_traj=args.no_pred_gripper_traj, no_unshuffle=args.no_unshuffle, use_gpt2_pretrained=args.use_gpt2_pretrained, share_query=args.share_query, attn_implementation=args.attn_implementation, use_fm=args.use_fm)
    model = model.float()
    model.vision_encoder.bfloat16()
    model.clip_model.requires_grad_(False)
    model.vision_encoder.requires_grad_(False)
    model = model.to(device)
    model._init_model_type()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt['model_state_dict']
    sd = {k[7:] if k.startswith('module.') else k: v for (k, v) in sd.items()}
    (missing, unexpected) = model.load_state_dict(sd, strict=False)
    model.eval()
    return (model, args, dict(missing=len(missing), unexpected=len(unexpected), missing_sample=list(missing)[:8], unexpected_sample=list(unexpected)[:8]))

def _cast_dtype(precision):
    if precision == 'bf16' or precision == 'amp_bf16':
        return torch.bfloat16
    if precision == 'fp16':
        return torch.float16
    return None

class DreamVLAPolicy:

    def __init__(self, model, args, device='cuda', continuous_gripper=True):
        self.model = model
        self.args = args
        self.device = device
        self.continuous_gripper = continuous_gripper
        self.cast_type = _cast_dtype(args.precision)
        self.history_len = args.sequence_length
        self.action_pred_steps = args.action_pred_steps
        self.max_steps = args.libero_eval_max_steps
        self.ensembling_temp = args.ensembling_temp
        self.gripper_width = args.gripper_width
        self.image_processor = model.image_processor
        import clip as _clip
        self._clip = _clip
        self.reset()

    def reset(self):
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.gripper_state = np.array([-1.0])
        self.all_time_actions = torch.zeros([self.max_steps, self.max_steps + self.action_pred_steps, 7]).to(self.device)
        self.raw_gripper_log = []
        self.cmd_gripper_log = []

    def _proc_img(self, arr):
        img = Image.fromarray(np.ascontiguousarray(arr))
        x = self.image_processor(img).unsqueeze(0)
        return x.unsqueeze(1).to(dtype=self.cast_type)

    def step(self, obs, goal, timestep):
        image_x = self._proc_img(obs['agentview_image'][::-1]).to(self.device)
        gripper = self._proc_img(obs['robot0_eye_in_hand_image']).to(self.device)
        text_x = self._clip.tokenize([goal], truncate=True).unsqueeze(1).to(self.device)
        state_pos = obs['robot0_eef_pos']
        state_ori = R.from_quat(obs['robot0_eef_quat']).as_euler('xyz', degrees=False)
        if not self.gripper_width:
            vec = np.concatenate([state_pos, state_ori, self.gripper_state])
        else:
            vec = np.concatenate([state_pos, state_ori, obs['robot0_gripper_qpos']])
        state = torch.from_numpy(vec.astype(np.float32))
        state = state.to(dtype=self.cast_type).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.img_queue.append(image_x)
            self.gripper_queue.append(gripper)
            self.state_queue.append(state)
            if len(self.text_queue) == 0:
                for _ in range(self.history_len):
                    self.text_queue.append(text_x)
            image_primary = torch.cat(list(self.img_queue), dim=1)
            image_wrist = torch.cat(list(self.gripper_queue), dim=1)
            state_seq = torch.cat(list(self.state_queue), dim=1)
            input_text_token = torch.cat(list(self.text_queue), dim=1)
            num_step = image_primary.shape[1]
            if num_step < self.history_len:
                pad = self.history_len - num_step
                input_image_primary = torch.cat([image_primary, image_primary[:, -1].repeat(1, pad, 1, 1, 1)], dim=1)
                input_image_wrist = torch.cat([image_wrist, image_wrist[:, -1].repeat(1, pad, 1, 1, 1)], dim=1)
                input_state = torch.cat([state_seq, state_seq[:, -1].repeat(1, pad, 1)], dim=1)
            else:
                (input_image_primary, input_image_wrist, input_state) = (image_primary, image_wrist, state_seq)
            (arm_action, gripper_action) = self.model(image_primary=input_image_primary, image_wrist=input_image_wrist, state=input_state, text_token=input_text_token, action=torch.zeros(1, self.history_len, 7).to(input_state.device), mode='test')[:2]
            selected_step = num_step - 1 if num_step < self.history_len else -1
            act = torch.concat((arm_action[:, selected_step], gripper_action[:, selected_step]), dim=-1)
            self.all_time_actions[timestep:timestep + 1, timestep:timestep + self.action_pred_steps] = act
            cur = self.all_time_actions[:, timestep]
            cur = cur[torch.all(cur != 0, axis=1)]
            w = np.exp(-self.ensembling_temp * np.arange(len(cur)))
            w = torch.from_numpy(w / w.sum()).to(self.device).unsqueeze(dim=1)
            action = (cur * w).sum(dim=0, keepdim=True)
            g_raw = action[:, 6:]
            self.raw_gripper_log.append(float(g_raw.flatten()[-1]))
            if self.continuous_gripper:
                g = torch.clamp((g_raw - 0.5) * 2.0, -1.0, 1.0)
            else:
                g = ((g_raw > 0.5).to(action.dtype) - 0.5) * 2.0
            action = torch.concat((action[:, :6], g), dim=-1)
            action = action.detach().cpu().float().numpy()[-1]
        self.gripper_state = np.array([action[-1]])
        self.cmd_gripper_log.append(float(action[-1]))
        return action

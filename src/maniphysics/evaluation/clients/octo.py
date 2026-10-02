from collections import deque
from typing import Optional, Sequence
import os
import jax
import matplotlib.pyplot as plt
import numpy as np
from octo.model.octo_model import OctoModel
import tensorflow as tf
from transforms3d.euler import euler2axangle
from simpler_env.utils.action.action_ensemble import ActionEnsembler

class OctoInference:

    def __init__(self, model: Optional[OctoModel]=None, dataset_id: Optional[str]=None, model_type: str='octo-base', policy_setup: str='widowx_bridge', horizon: int=2, pred_action_horizon: int=4, exec_horizon: int=1, image_size: int=256, action_scale: float=1.0, init_rng: int=0) -> None:
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        if policy_setup == 'widowx_bridge':
            dataset_id = 'bridge_dataset' if dataset_id is None else dataset_id
            action_ensemble = True
            action_ensemble_temp = 0.0
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == 'google_robot':
            dataset_id = 'fractal20220817_data' if dataset_id is None else dataset_id
            action_ensemble = True
            action_ensemble_temp = 0.0
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(f'Policy setup {policy_setup} not supported for octo models.')
        self.policy_setup = policy_setup
        self.dataset_id = dataset_id
        if model is not None:
            (self.tokenizer, self.tokenizer_kwargs) = (None, None)
            self.model = model
            self.action_mean = self.model.dataset_statistics[dataset_id]['action']['mean']
            self.action_std = self.model.dataset_statistics[dataset_id]['action']['std']
        elif model_type in ['octo-base', 'octo-small']:
            self.model_type = f'hf://rail-berkeley/{model_type}'
            (self.tokenizer, self.tokenizer_kwargs) = (None, None)
            self.model = OctoModel.load_pretrained(self.model_type)
            self.action_mean = self.model.dataset_statistics[dataset_id]['action']['mean']
            self.action_std = self.model.dataset_statistics[dataset_id]['action']['std']
        else:
            raise NotImplementedError()
        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon
        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = action_ensemble_temp
        self.rng = jax.random.PRNGKey(init_rng)
        for _ in range(5):
            (self.rng, _key) = jax.random.split(self.rng)
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.task = None
        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(self.pred_action_horizon, self.action_ensemble_temp)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = tf.image.resize(image, size=(self.image_size, self.image_size), method='lanczos3', antialias=True)
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def _obtain_image_history_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        images = np.stack(self.image_history, axis=0)
        horizon = len(self.image_history)
        pad_mask = np.ones(horizon, dtype=np.float64)
        pad_mask[:horizon - min(horizon, self.num_image_history)] = 0
        return (images, pad_mask)

    def reset(self, task_description: str) -> None:
        self.task = self.model.create_tasks(texts=[task_description])
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(self, image: np.ndarray, task_description: Optional[str]=None, *args, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)
        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        (images, pad_mask) = self._obtain_image_history_and_mask()
        (images, pad_mask) = (images[None], pad_mask[None])
        (self.rng, key) = jax.random.split(self.rng)
        input_observation = {'image_primary': images, 'pad_mask': pad_mask}
        norm_raw_actions = self.model.sample_actions(input_observation, self.task, rng=key)
        raw_actions = norm_raw_actions * self.action_std[None] + self.action_mean[None]
        raw_actions = raw_actions[0]
        assert raw_actions.shape == (self.pred_action_horizon, 7)
        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)
            raw_actions = raw_actions[None]
        raw_action = {'world_vector': np.array(raw_actions[0, :3]), 'rotation_delta': np.array(raw_actions[0, 3:6]), 'open_gripper': np.array(raw_actions[0, 6:7])}
        action = {}
        action['world_vector'] = raw_action['world_vector'] * self.action_scale
        action_rotation_delta = np.asarray(raw_action['rotation_delta'], dtype=np.float64)
        (roll, pitch, yaw) = action_rotation_delta
        (action_rotation_ax, action_rotation_angle) = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action['rot_axangle'] = action_rotation_axangle * self.action_scale
        if self.policy_setup == 'google_robot':
            current_gripper_action = raw_action['open_gripper']
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action
            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0
            action['gripper'] = relative_gripper_action
        elif self.policy_setup == 'widowx_bridge':
            _g = np.asarray(raw_action['open_gripper'], dtype=np.float64)
            action['gripper'] = 2.0 * np.clip(_g, 0.0, 1.0) - 1.0 if os.environ.get('MANIPHYS_CONTINUOUS_GRIPPER') else 2.0 * (_g > 0.5) - 1.0
        action['terminate_episode'] = np.array([0.0])
        return (raw_action, action)

    def visualize_epoch(self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'grasp']
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
        figure_layout = [['image'] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({'font.size': 12})
        (fig, axs) = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])
        pred_actions = np.array([np.concatenate([a['world_vector'], a['rotation_delta'], a['open_gripper']], axis=-1) for a in predicted_raw_actions])
        for (action_dim, action_label) in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label='predicted action')
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel('Time in one episode')
        axs['image'].imshow(img_strip)
        axs['image'].set_xlabel('Time in one episode (subsampled)')
        plt.legend()
        plt.savefig(save_path)

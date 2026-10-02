from maniphysics.paths import release_root
import numpy as np
__import__('mani_skill2_real2sim.envs')
from mani_skill2_real2sim.envs.custom_scenes.put_on_in_scene import PutOnBridgeInSceneEnv
from maniphysics.bench.sapien.mixin import PhysConstraintMixin

class ManiPhysPutOnTowelBase(PhysConstraintMixin, PutOnBridgeInSceneEnv):
    SOURCE_ID = None
    INSTRUCTION = None
    SOURCE_QUAT = (1, 0, 0, 0)
    XY_HALF = 0.075
    CUSTOM_XY = None

    def __init__(self, **kwargs):
        if self.CUSTOM_XY is not None:
            xy_configs = [np.array(c) for c in self.CUSTOM_XY]
        else:
            xy_center = np.array([-0.16, 0.0])
            grid_pos = (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1) * self.XY_HALF + xy_center
            xy_configs = [np.array([p1, p2]) for (i, p1) in enumerate(grid_pos) for (j, p2) in enumerate(grid_pos) if i != j]
        quat_configs = [np.array([list(self.SOURCE_QUAT), [1, 0, 0, 0]])]
        kwargs.setdefault('asset_root', str(release_root() / 'bench/assets'))
        kwargs.setdefault('model_json', 'registry.json')
        super().__init__(source_obj_name=self.SOURCE_ID, target_obj_name='table_cloth_generated_shorter', xy_configs=xy_configs, quat_configs=quat_configs, **kwargs)

    def evaluate(self, success_require_src_completely_on_target=False, **kwargs):
        if getattr(self, 'episode_source_obj_bbox_world', None) is not None:
            self.episode_source_obj_bbox_world = np.abs(self.episode_source_obj_bbox_world)
        if getattr(self, 'episode_target_obj_bbox_world', None) is not None:
            self.episode_target_obj_bbox_world = np.abs(self.episode_target_obj_bbox_world)
        return super().evaluate(success_require_src_completely_on_target=success_require_src_completely_on_target, **kwargs)

    def get_language_instruction(self, **kwargs):
        return self.INSTRUCTION
_TALL_OBJ_XY = [[[-0.1, -0.075], [-0.1, 0.075]], [[-0.1, 0.075], [-0.1, -0.075]]]
_DIAG_XY = [[[-0.235, -0.075], [-0.085, 0.075]], [[-0.085, 0.075], [-0.235, -0.075]], [[-0.235, 0.075], [-0.085, -0.075]], [[-0.085, -0.075], [-0.235, 0.075]]]

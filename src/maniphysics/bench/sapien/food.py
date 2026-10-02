import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from maniphysics.bench.sapien.tasks import ManiPhysPutOnTowelBase, _DIAG_XY
from mani_skill2_real2sim.utils.registration import register_env
FOOD_TASKS = [('Walnut', 'walnut-in-shell', 'put the walnut on the towel', (1, 0, 0, 0), None, 'put the walnut on the towel'), ('CherryTomatoW30L40', 'cherry-tomato-w30l40mm', 'put the cherry tomato on the towel', (1, 0, 0, 0), None, 'gently put the fragile cherry tomato on the towel'), ('Potato', 'boiled-potato', 'put the potato on the towel', (1, 0, 0, 0), None, 'gently put the soft potato on the towel'), ('Strawberry', 'strawberry', 'put the strawberry on the towel', (1, 0, 0, 0), None, 'gently put the delicate strawberry on the towel'), ('Plum', 'plum', 'put the plum on the towel', (1, 0, 0, 0), None, 'gently put the soft plum on the towel')]
for (_sfx, _mid, _instr, _quat, _cxy, _hint) in FOOD_TASKS:
    for (_tag, _text) in (('', _instr), ('Hint', _hint)):
        _cls = type(f'ManiPhys{_sfx}{_tag}Env', (ManiPhysPutOnTowelBase,), dict(SOURCE_ID=_mid, INSTRUCTION=_text, SOURCE_QUAT=_quat, CUSTOM_XY=_cxy))
        globals()[_cls.__name__] = _cls
        register_env(f'ManiPhys{_sfx}{_tag}-v0', max_episode_steps=120)(_cls)

@register_env('ManiPhysEgg-v0', max_episode_steps=120)
class ManiPhysEggEnv(ManiPhysPutOnTowelBase):
    SOURCE_ID = 'chicken-egg'
    INSTRUCTION = 'put the egg on the towel'
    SOURCE_QUAT = tuple(euler2quat(0, np.pi / 2, 0))
    CUSTOM_XY = _DIAG_XY
    CARTON_ID = 'egg-carton-molded-pulp'
    _SEAT_DZ = 0.004

    def _load_model(self):
        super()._load_model()
        ent = self.model_db.get(self.CARTON_ID, {})
        self._carton = self._build_actor_helper(self.CARTON_ID, self._scene, density=ent.get('density', 300), physical_material=self._scene.create_physical_material(0.6, 0.6, 0.0), root_dir=self.asset_root)
        self._carton.name = self.CARTON_ID
        bb = ent.get('bbox') or {}
        self._carton_h = bb.get('max', [0, 0, 0.03])[2] - bb.get('min', [0, 0, 0])[2] or 0.03

    def _initialize_actors(self):
        super()._initialize_actors()
        egg = self.episode_source_obj
        (x, y) = egg.pose.p[:2]
        self._carton.set_pose(sapien.Pose([x, y, self.scene_table_height + self._carton_h / 2 + 0.001]))
        egg.set_pose(sapien.Pose([x, y, self.scene_table_height + self._carton_h - self._SEAT_DZ + (egg.pose.p[2] - self.scene_table_height)], egg.pose.q))
        self._settle(0.4)

@register_env('ManiPhysEggHint-v0', max_episode_steps=120)
class ManiPhysEggHintEnv(ManiPhysEggEnv):
    INSTRUCTION = 'gently put the fragile egg on the towel'
A_ENVS = ['ManiPhysWalnut-v0', 'ManiPhysEgg-v0', 'ManiPhysCherryTomatoW30L40-v0']
B_ENVS = ['ManiPhysPotato-v0', 'ManiPhysStrawberry-v0', 'ManiPhysPlum-v0']
FOOD_ENVS = A_ENVS + B_ENVS
HINT_ENVS = [e.replace('-v0', 'Hint-v0') for e in FOOD_ENVS]

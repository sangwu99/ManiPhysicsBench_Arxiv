import os
import numpy as np
from maniphysics.bench.libero.gripper import register as _register_gripper, REGISTERED_NAME, ABSOLUTE_NAME
from maniphysics.bench.libero.logger import ManiPhysMujocoMixin
ENV_SWITCH = 'MANIPHYS_CONTINUOUS_GRIPPER'
STOCK_NAME = 'PandaGripper'
_FALSY = {'', '0', 'false', 'off', 'no', 'none', 'binary', 'stock'}
_ALIASES = {'1': ABSOLUTE_NAME, 'true': ABSOLUTE_NAME, 'on': ABSOLUTE_NAME, 'yes': ABSOLUTE_NAME, 'abs': ABSOLUTE_NAME, 'absolute': ABSOLUTE_NAME, 'rate': REGISTERED_NAME, 'integrate': REGISTERED_NAME, 'continuous': REGISTERED_NAME}
_INSTALLED = {'done': False, 'patched': [], 'gripper': None}

def resolve_gripper(default=ABSOLUTE_NAME):
    v = os.environ.get(ENV_SWITCH)
    if v is None:
        return default
    v = v.strip().lower()
    if v in _FALSY:
        return STOCK_NAME
    return _ALIASES.get(v, ABSOLUTE_NAME)

def install_gripper(verbose=True):
    _register_gripper()
    from robosuite.models import grippers as _g
    from robosuite.models.grippers.panda_gripper import PandaGripper
    name = resolve_gripper(default=STOCK_NAME)
    _g.GRIPPER_MAPPING[STOCK_NAME] = PandaGripper if name == STOCK_NAME else _g.GRIPPER_MAPPING[name]
    if verbose and _INSTALLED.get('gripper') != name:
        print(f"[gripper] GRIPPER_MAPPING['{STOCK_NAME}'] → {name} ({('CONTINUOUS' if name != STOCK_NAME else 'BINARY(stock np.sign)')}); {ENV_SWITCH}={os.environ.get(ENV_SWITCH)!r}", flush=True)
    _INSTALLED['gripper'] = name
    return name

def install():
    install_gripper()
    from libero.libero.envs import bddl_base_domain as _bd
    if _INSTALLED['done']:
        return _INSTALLED['patched']
    for (name, cls) in list(_bd.TASK_MAPPING.items()):
        if issubclass(cls, ManiPhysMujocoMixin):
            continue
        _bd.TASK_MAPPING[name] = type(cls.__name__, (ManiPhysMujocoMixin, cls), {})
        _INSTALLED['patched'].append(name)
    _INSTALLED['done'] = True
    return _INSTALLED['patched']

def make_env(task_id=0, suite_name='libero_spatial', bddl_file=None, continuous_gripper=True, camera_size=128, gripper_types=None, **maniphys_kwargs):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    install()
    if bddl_file is None:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        task = suite.get_task(task_id)
        bddl_file = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
    caller_default = gripper_types or (ABSOLUTE_NAME if continuous_gripper else STOCK_NAME)
    kw = dict(bddl_file_name=bddl_file, camera_heights=camera_size, camera_widths=camera_size, gripper_types=resolve_gripper(caller_default), **maniphys_kwargs)
    return OffScreenRenderEnv(**kw)

def gripper_spec(env):
    e = env.env if hasattr(env, 'env') else env
    (sim, robot) = (e.sim, e.robots[0])
    ids = list(robot._ref_joint_gripper_actuator_indexes)
    return dict(actuator_ids=ids, kp=[float(sim.model.actuator_gainprm[i][0]) for i in ids], forcerange=[np.array(sim.model.actuator_forcerange[i]).tolist() for i in ids], ctrlrange=[np.array(sim.model.actuator_ctrlrange[i]).tolist() for i in ids], gripper_class=type(robot.gripper).__name__)

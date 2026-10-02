from maniphysics.bench.sapien.tasks import ManiPhysPutOnTowelBase, _TALL_OBJ_XY
from mani_skill2_real2sim.utils.registration import register_env
CONTAINER_TASKS = [('Yakult', 'yakult-bottle', 'put the yakult bottle on the towel', (1, 0, 0, 0), _TALL_OBJ_XY, 'gently put the yakult bottle on the towel'), ('PaperCup2oz', 'paper-cup-2oz', 'put the small paper cup on the towel', (1, 0, 0, 0), None, 'gently put the small paper cup on the towel'), ('PortionCup', 'portion-cup-1oz-pp', 'put the plastic sauce cup on the towel', (1, 0, 0, 0), None, 'gently put the plastic sauce cup on the towel')]
for (_sfx, _mid, _instr, _quat, _cxy, _hint) in CONTAINER_TASKS:
    for (_tag, _text) in (('', _instr), ('Hint', _hint)):
        _cls = type(f'ManiPhys{_sfx}{_tag}Env', (ManiPhysPutOnTowelBase,), dict(SOURCE_ID=_mid, INSTRUCTION=_text, SOURCE_QUAT=_quat, CUSTOM_XY=_cxy))
        globals()[_cls.__name__] = _cls
        register_env(f'ManiPhys{_sfx}{_tag}-v0', max_episode_steps=120)(_cls)

from maniphysics.paths import release_root, runtime_root
import os
import re
from pathlib import Path
MPB = release_root()
BDDL_DIR = runtime_root() / 'bddl'
TASKS = [('Walnut', 'walnut-in-shell', 'pick up the walnut and place it in the basket', 'A'), ('Egg', 'chicken-egg', 'pick up the egg and place it in the basket', 'A'), ('CherryTomatoW30L40', 'cherry-tomato-w30l40mm', 'pick up the cherry tomato and place it in the basket', 'A'), ('Potato', 'boiled-potato', 'pick up the potato and place it in the basket', 'B'), ('Strawberry', 'strawberry', 'pick up the strawberry and place it in the basket', 'B'), ('Plum', 'plum', 'pick up the plum and place it in the basket', 'B'), ('PaperCup', 'paper-cup-takeaway', 'pick up the paper cup and place it in the basket', 'C2'), ('Yakult', 'yakult-bottle', 'pick up the yakult bottle and place it in the basket', 'C2'), ('PortionCup', 'portion-cup-1oz-pp', 'pick up the sauce cup and place it in the basket', 'C2')]
TASK_BY_SUFFIX = {t[0]: t for t in TASKS}
A_SUFFIXES = ['Walnut', 'Egg', 'CherryTomatoW30L40']
B_SUFFIXES = ['Potato', 'Strawberry', 'Plum']
FOOD_SUFFIXES = A_SUFFIXES + B_SUFFIXES
C_SUFFIXES = ['PaperCup', 'PortionCup', 'Yakult']
SEAT_ON = {'Egg': 'egg-carton-molded-pulp'}
_TMPL = '(define (problem LIBERO_Tabletop_Manipulation)\n  (:domain robosuite)\n  (:language {language})\n    (:regions\n      (obj_region\n          (:target main_table)\n          (:ranges (\n              ({ox0} {oy0} {ox1} {oy1})\n            )\n          )\n      )\n      (plate_region\n          (:target main_table)\n          (:ranges (\n              (-0.10 0.27 -0.08 0.29)\n            )\n          )\n      )\n{extra_regions}    )\n\n  (:fixtures\n    main_table - table\n  )\n\n  (:objects\n{obj_decls}    plate_1 - plate\n  )\n\n  (:obj_of_interest\n    {obj_inst}\n    plate_1\n  )\n\n  (:init\n{inits}    (On plate_1 main_table_plate_region)\n  )\n\n  (:goal\n    (And (On {obj_inst} plate_1))\n  )\n\n)\n'
_DISTRACTOR_POOL = ['salad_dressing', 'cream_cheese', 'milk', 'tomato_sauce', 'butter']
DISTRACTORS = _DISTRACTOR_POOL if os.environ.get('MANIPHYS_DISTRACTORS', '') not in ('', '0') else []
_OTHER_RANGES = [(0.025, -0.125, 0.075, -0.075), (-0.175, 0.035, -0.125, 0.085), (0.075, -0.225, 0.125, -0.175), (0.125, 0.005, 0.175, 0.055), (-0.225, -0.105, -0.175, -0.055)]
_TMPL_OBJECT = '(define (problem LIBERO_Floor_Manipulation)\n  (:domain robosuite)\n  (:language {language})\n    (:regions\n      (bin_region\n          (:target floor)\n          (:ranges (\n              (-0.01 0.25 0.01 0.27)\n            )\n          )\n      )\n      (target_object_region\n          (:target floor)\n          (:ranges (\n              (-0.145 -0.265 -0.095 -0.215)\n            )\n          )\n      )\n{other_regions}      (contain_region\n          (:target basket_1)\n      )\n    )\n\n  (:fixtures\n    floor - floor\n  )\n\n  (:objects\n{obj_decls}    basket_1 - basket\n  )\n\n  (:obj_of_interest\n    {obj_inst}\n    basket_1\n  )\n\n  (:init\n{inits}    (On basket_1 floor_bin_region)\n  )\n\n  (:goal\n    (And (In {obj_inst} basket_1_contain_region))\n  )\n\n)\n'

def _atomic_write(path, xml):
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(xml)
    os.replace(tmp, path)
    return path

def make_bddl_object_suite(obj_type, language=None, out_dir=None, seat_on=None):
    obj_inst = f'{obj_type}_1'
    nice = obj_type.replace('maniphys_', '').replace('ycb_', '').replace('_', ' ')
    decls = [f'    {obj_inst} - {obj_type}\n']
    inits = [f'    (On {obj_inst} floor_target_object_region)\n']
    if seat_on:
        seat_inst = f'{seat_on}_1'
        decls.append(f'    {seat_inst} - {seat_on}\n')
        inits = [f'    (On {seat_inst} floor_target_object_region)\n', f'    (On {obj_inst} {seat_inst})\n']
    (others, regions) = ([], [])
    for (i, dz) in enumerate([d for d in DISTRACTORS if d != obj_type]):
        r = _OTHER_RANGES[i]
        regions.append(f'      (other_object_region_{i}\n          (:target floor)\n          (:ranges (\n              ({r[0]} {r[1]} {r[2]} {r[3]})\n            )\n          )\n      )\n')
        decls.append(f'    {dz}_1 - {dz}\n')
        others.append(f'    (On {dz}_1 floor_other_object_region_{i})\n')
    inits += others
    xml = _TMPL_OBJECT.format(language=language or f'pick up the {nice} and place it in the basket', obj_inst=obj_inst, obj_decls=''.join(decls), inits=''.join(inits), other_regions=''.join(regions))
    out_dir = Path(out_dir or BDDL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'maniphys_obj_{obj_type}.bddl'
    return _atomic_write(path, xml)
CONTROL_STOCK_OBJECTS = ['alphabet_soup', 'milk', 'tomato_sauce', 'butter', 'cream_cheese']

def make_control_env(stock_obj='alphabet_soup', log_dir=None, camera_size=128, **kw):
    from maniphysics.bench.libero import env as E
    from maniphysics.bench.libero import gripper as G
    bddl = make_bddl_object_suite(stock_obj, language=f"pick up the {stock_obj.replace('_', ' ')} and place it in the basket")
    return E.make_env(bddl_file=str(bddl), gripper_types=G.ABSOLUTE_NAME, camera_size=camera_size, maniphys_model_id=None, maniphys_obj_name=f'{stock_obj}_1', maniphys_log_dir=log_dir, **kw)

def make_bddl(obj_type, language=None, out_dir=None, region=(-0.02, -0.01, 0.02, 0.01), pellets=0, pellet_type='maniphys_pellet', seat_on=None):
    obj_inst = f'{obj_type}_1'
    nice = obj_type.replace('maniphys_', '').replace('ycb_', '').replace('_', ' ')
    decls = [f'    {obj_inst} - {obj_type}\n']
    inits = [f'    (On {obj_inst} main_table_obj_region)\n']
    extra = ''
    if seat_on:
        seat_inst = f'{seat_on}_1'
        decls.append(f'    {seat_inst} - {seat_on}\n')
        inits = [f'    (On {seat_inst} main_table_obj_region)\n', f'    (On {obj_inst} {seat_inst})\n']
    if pellets:
        names = ' '.join((f'{pellet_type}_{i + 1}' for i in range(pellets)))
        decls.append(f'    {names} - {pellet_type}\n')
        extra = '      (pellet_region\n          (:target main_table)\n          (:ranges (\n              (0.10 -0.22 0.24 -0.06)\n            )\n          )\n      )\n'
        for i in range(pellets):
            inits.append(f'    (On {pellet_type}_{i + 1} main_table_pellet_region)\n')
    xml = _TMPL.format(language=language or f'pick up the {nice} and place it on the plate', obj_inst=obj_inst, obj_decls=''.join(decls), inits=''.join(inits), extra_regions=extra, ox0=region[0], oy0=region[1], ox1=region[2], oy1=region[3])
    out_dir = Path(out_dir or BDDL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'maniphys_{obj_type}.bddl'
    return _atomic_write(path, xml)

def build_all(pellets_for_containers=6):
    from maniphysics.bench.libero import objects as O
    out = {}
    for (suffix, model_id, lang, axis) in TASKS:
        (key, _) = O.register_libero_object(model_id)
        n_pel = pellets_for_containers if axis == 'E' else 0
        if n_pel:
            O.register_pellet()
        seat = SEAT_ON.get(suffix)
        seat_key = O.register_libero_object(seat)[0] if seat else None
        path = make_bddl(key, language=lang, pellets=n_pel, seat_on=seat_key)
        out[suffix] = (model_id, str(path), axis)
    return out

def make_maniphys_env(suffix_or_model='Egg', log_dir=None, continuous_gripper=True, absolute_gripper=True, camera_size=128, pellets=None, scale=1.0, **kw):
    from maniphysics.bench.libero import objects as O
    from maniphysics.bench.libero import env as E
    from maniphysics.bench.libero import gripper as G
    if suffix_or_model in TASK_BY_SUFFIX:
        (_, model_id, lang, axis) = TASK_BY_SUFFIX[suffix_or_model]
    else:
        (model_id, lang, axis) = (suffix_or_model, None, '-')
    if scale != 1.0:
        cname = ''.join((p.capitalize() for p in re.sub('[^0-9a-zA-Z]+', '_', model_id).split('_')))
        (key, _) = O.register_libero_object(model_id, class_name=f'{cname}S{int(round(scale * 100))}', scale=scale)
        return E.make_env(bddl_file=str(make_bddl_object_suite(key, language=lang)), gripper_types=G.ABSOLUTE_NAME, camera_size=camera_size, maniphys_model_id=model_id, maniphys_obj_name=f'{key}_1', maniphys_log_dir=log_dir, **kw)
    (key, _) = O.register_libero_object(model_id)
    n_pel = (6 if axis == 'E' else 0) if pellets is None else pellets
    if n_pel:
        O.register_pellet()
    seat = SEAT_ON.get(suffix_or_model)
    seat_key = O.register_libero_object(seat)[0] if seat else None
    if suffix_or_model in TASK_BY_SUFFIX:
        bddl = make_bddl_object_suite(key, language=lang, seat_on=seat_key)
    else:
        bddl = make_bddl(key, language=lang, pellets=n_pel, seat_on=seat_key)
    gt = G.ABSOLUTE_NAME if absolute_gripper else G.REGISTERED_NAME if continuous_gripper else 'PandaGripper'
    return E.make_env(bddl_file=str(bddl), gripper_types=gt, camera_size=camera_size, maniphys_model_id=model_id, maniphys_obj_name=f'{key}_1', maniphys_log_dir=log_dir, **kw)

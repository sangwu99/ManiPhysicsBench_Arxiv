from maniphysics.paths import release_root, runtime_root
from maniphysics.bench.obj_parts import split_obj_parts
import json
import re
from pathlib import Path
import numpy as np
MPB = release_root()
SAPIEN_MODELS = release_root() / 'bench/assets/models'
REGISTRY = release_root() / 'bench/assets/registry.json'
OUT_DIR = runtime_root() / 'libero_assets'
_HEAD = '<mujoco model="{name}">\n  <asset>\n{meshes}  </asset>\n  <worldbody>\n    <body>\n      <body name="object">\n{geoms}      </body>\n      <site rgba="0 0 0 0" size="0.005" pos="0 0 {zmin}" name="bottom_site"/>\n      <site rgba="0 0 0 0" size="0.005" pos="0 0 {zmax}" name="top_site"/>\n      <site rgba="0 0 0 0" size="0.005" pos="{hr} {hr} 0" name="horizontal_radius_site"/>\n    </body>\n  </worldbody>\n</mujoco>\n'

def convert_visual(model_id, out_dir, tex_max=1024):
    src = next((SAPIEN_MODELS / model_id / f for f in ('textured.glb', 'textured.dae', 'textured.obj') if (SAPIEN_MODELS / model_id / f).exists()), None)
    if src is None:
        return (None, None, None)
    import trimesh
    scene = trimesh.load(str(src), process=False)
    mesh = scene.to_geometry() if hasattr(scene, 'geometry') else scene
    out_dir = Path(out_dir)
    (tex_name, rgba) = (None, None)
    vis = getattr(mesh, 'visual', None)
    mat = getattr(vis, 'material', None)
    img = None
    if mat is not None:
        img = getattr(mat, 'baseColorTexture', None) or getattr(mat, 'image', None)
    uv = getattr(vis, 'uv', None) if vis is not None else None
    if img is not None and uv is not None and len(uv):
        im = img.convert('RGB')
        if max(im.size) > tex_max:
            im = im.resize((min(im.width, tex_max), min(im.height, tex_max)))
        tex_name = 'texture.png'
        im.save(out_dir / tex_name)
    else:
        fac = getattr(mat, 'baseColorFactor', None) if mat is not None else None
        if fac is not None:
            f = np.asarray(fac, dtype=float)
            f = f / 255.0 if f.max() > 1.0 else f
            rgba = ' '.join((f'{v:.3f}' for v in list(f[:3]) + [1.0]))
    obj_name = 'visual_tex.obj'
    mesh.export(out_dir / obj_name, include_texture=bool(tex_name))
    return (obj_name, tex_name, rgba)
MESH_ALIAS = {}
RESCALE = {}
PRE_ROTATE = {'chicken-egg': (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)}

def _quat_to_mat(q):
    (w, x, y, z) = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def build_mjcf(model_id, rgba='0.93 0.90 0.80 1', scale=1.0, out_dir=None, max_parts=None, container_site=False):
    reg = json.loads(REGISTRY.read_text())
    entry = reg.get(model_id, {})
    src = SAPIEN_MODELS / MESH_ALIAS.get(model_id, model_id) / 'collision.obj'
    if not src.exists():
        raise FileNotFoundError(f'Missing mesh: {src}')
    out_dir = Path(out_dir or OUT_DIR) / model_id
    parts = split_obj_parts(src, out_dir)
    sc = np.asarray(RESCALE.get(model_id, (1.0, 1.0, 1.0)), dtype=float) * float(scale)
    bbox = entry.get('bbox') or {}
    mn = np.array(bbox.get('min', [-0.02, -0.02, -0.02])) * sc
    mx = np.array(bbox.get('max', [0.02, 0.02, 0.02])) * sc
    fs = entry.get('static_friction', 0.5)
    density = entry.get('density', 1000)
    q = PRE_ROTATE.get(model_id)
    (gq, smn, smx) = ('', mn, mx)
    if q is not None:
        R = _quat_to_mat(q)
        cor = np.array([[a, b, c] for a in (mn[0], mx[0]) for b in (mn[1], mx[1]) for c in (mn[2], mx[2])])
        rc = cor @ R.T
        (smn, smx) = (rc.min(axis=0), rc.max(axis=0))
        gq = ' quat="{:.9f} {:.9f} {:.9f} {:.9f}"'.format(*q)
    (meshes, geoms) = ([], [])
    if model_id in BOX_SHAPED:
        half = (mx - mn) / 2.0
        ctr = (mx + mn) / 2.0
        geoms.append(f'        <geom type="box" size="{half[0]:.5f} {half[1]:.5f} {half[2]:.5f}" pos="{ctr[0]:.5f} {ctr[1]:.5f} {ctr[2]:.5f}"{gq} name="{model_id}_g0" group="0"\n              density="{density}" friction="{fs} 0.05 0.002" condim="4"\n              solimp="0.999 0.9999 0.0005" solref="0.0005 1"/>\n')
        parts = []
    for (i, p) in enumerate(parts):
        mn_ = f'{model_id}_m{i}'
        meshes.append(f'    <mesh file="{p}" name="{mn_}" scale="{sc[0]} {sc[1]} {sc[2]}"/>\n')
        geoms.append(f'        <geom type="mesh" mesh="{mn_}"{gq} name="{model_id}_g{i}" group="0"\n              density="{density}" friction="{fs} 0.05 0.002" condim="4"\n              solimp="0.999 0.9999 0.0005" solref="0.0005 1"/>\n')
    (assets, vis_obj, tex, glb_rgba) = ('', None, None, None)
    (vis_obj, tex, glb_rgba) = convert_visual(model_id, out_dir)
    if vis_obj is None:
        _src_lines = Path(src).read_text(errors='ignore').splitlines()
        _merged = [ln for ln in _src_lines if ln[:2] not in ('o ', 'g ') and (ln.startswith('v ') or ln.startswith('f '))]
        (Path(out_dir) / 'visual.obj').write_text('\n'.join(_merged) + '\n')
        vis_obj = 'visual.obj'
    meshes.append(f'    <mesh file="{vis_obj}" name="{model_id}_vism" scale="{sc[0]} {sc[1]} {sc[2]}"/>\n')
    if tex:
        assets = f'    <texture type="2d" name="{model_id}_tex" file="{tex}"/>\n    <material name="{model_id}_mat" texture="{model_id}_tex" specular="0.2" shininess="0.3"/>\n'
        vis_attr = f'material="{model_id}_mat"'
    else:
        vis_attr = f'rgba="{glb_rgba or rgba}"'
    meshes.append(assets)
    geoms.append(f'        <geom type="mesh" mesh="{model_id}_vism"{gq} name="{model_id}_vis" group="1" conaffinity="0" contype="0" {vis_attr}/>\n')
    if container_site:
        rin = float(max(abs(mx[0]), abs(mx[1]))) * 0.45
        zin = float(mx[2]) * 0.3
        n = int(container_site) if container_site is not True else 6
        for i in range(n):
            th = 2 * np.pi * i / max(n, 1)
            (px, py) = (rin * 0.55 * np.cos(th), rin * 0.55 * np.sin(th))
            geoms.append(f'        <site name="inside_region_{i}" type="box" size="0.004 0.004 0.002" pos="{px:.4f} {py:.4f} {zin}" quat="1 0 0 0" rgba="0 0 0 0"/>\n')
    xml = _HEAD.format(name=model_id, meshes=''.join(meshes), geoms=''.join(geoms), zmin=float(smn[2]), zmax=float(smx[2]), hr=float(max(abs(smx[0]), abs(smx[1]))))
    path = out_dir / f'{model_id}.xml'
    path.write_text(xml)
    return path
_PELLET = '<mujoco model="maniphys_pellet">\n  <worldbody>\n    <body>\n      <body name="object">\n        <geom type="sphere" size="{r}" name="maniphys_pellet_g0" group="0"\n              density="600" friction="0.6 0.005 0.0001"\n              solimp="0.998 0.998 0.001" solref="0.001 1" rgba="0.85 0.72 0.35 1"/>\n      </body>\n      <site rgba="0 0 0 0" size="0.002" pos="0 0 -{r}" name="bottom_site"/>\n      <site rgba="0 0 0 0" size="0.002" pos="0 0 {r}" name="top_site"/>\n      <site rgba="0 0 0 0" size="0.002" pos="{r} {r} 0" name="horizontal_radius_site"/>\n    </body>\n  </worldbody>\n</mujoco>\n'

def register_pellet(radius=0.006, out_dir=None):
    from robosuite.models.objects import MujocoXMLObject
    from libero.libero.envs import base_object as _bo
    key = 'maniphys_pellet'
    if key in _bo.OBJECTS_DICT and getattr(_bo.OBJECTS_DICT[key], '_maniphys', False):
        return (key, _bo.OBJECTS_DICT[key])
    d = Path(out_dir or OUT_DIR) / key
    d.mkdir(parents=True, exist_ok=True)
    p = d / f'{key}.xml'
    p.write_text(_PELLET.format(r=radius))

    def __init__(self, name=key, obj_name=key, joints=[dict(type='free', damping='0.0005')], _p=str(p)):
        MujocoXMLObject.__init__(self, _p, name=name, joints=joints, obj_type='all', duplicate_collision_geoms=False)
        self.category_name = key
        self.rotation = (0, 0)
        self.rotation_axis = 'z'
        self.object_properties = {'vis_site_names': {}}
    cls = type('ManiphysPellet', (MujocoXMLObject,), {'__init__': __init__, '_maniphys': True})
    _bo.OBJECTS_DICT[key] = cls
    return (key, cls)
BOX_SHAPED = set()
CONTAINERS = set()

def register_libero_object(model_id, class_name=None, **kw):
    from robosuite.models.objects import MujocoXMLObject
    from libero.libero.envs import base_object as _bo
    cname = class_name or ''.join((p.capitalize() for p in model_id.split('_')))
    key = '_'.join(re.sub('([A-Z0-9])', ' \\1', cname).split()).lower()
    if key in _bo.OBJECTS_DICT and getattr(_bo.OBJECTS_DICT[key], '_maniphys', False):
        return (key, _bo.OBJECTS_DICT[key])
    kw.setdefault('container_site', model_id in CONTAINERS)
    path = build_mjcf(model_id, **kw)

    def __init__(self, name=model_id, obj_name=model_id, joints=[dict(type='free', damping='0.05')], _p=str(path)):
        MujocoXMLObject.__init__(self, _p, name=name, joints=joints, obj_type='all', duplicate_collision_geoms=False)
        self.category_name = model_id
        self.rotation = (0, 0)
        self.rotation_axis = 'z'
        self.object_properties = {'vis_site_names': {}}
    cls = type(cname, (MujocoXMLObject,), {'__init__': __init__, '_maniphys': True})
    _bo.OBJECTS_DICT[key] = cls
    return (key, cls)

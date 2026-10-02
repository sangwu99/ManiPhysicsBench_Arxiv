import atexit
import os
from collections import OrderedDict
from pathlib import Path
import numpy as np
import sapien.core as sapien
from transforms3d.quaternions import quat2mat, qmult, qinverse
from mani_skill2_real2sim.utils.sapien_utils import get_pairwise_contacts
from maniphysics.bench.pad import PAD_C0, PAD_SPEC, SILICONE_GAMMA
from maniphysics.bench.logger import TrajBuffer
from maniphysics.bench.properties import PhysPropertyRegistry
_GRASP_DEBOUNCE = 3
_ENV_COUNTER = [0]
_PENDING = []

def _close_sign(gr, cfg):
    if gr is None or cfg is None or (not hasattr(gr, 'joints')):
        return None
    (lo, hi) = (getattr(cfg, 'lower', None), getattr(cfg, 'upper', None))
    if lo is None or hi is None:
        return None
    return [float(np.sign(hi - lo))] * len(gr.joints)

def _write_entry(e):
    if len(e['buf']) == 0:
        return
    st = e['state']
    meta = dict(episode=e['episode'], model_id=e['model_id'], robot_uid=e['robot_uid'], control_freq=e['control_freq'], sim_freq=e['sim_freq'], props=e['props'], contact_schema='normal-dyad', contact_every=1, physics_dt=1.0 / e['sim_freq'], violations={k: st[k] for k in ('toppled', 'slipped', 'rot_slipped', 'spilled')}, first_violation=st['first_violation'], final_tilt_deg=st.get('last_tilt'), success=e.get('success'), episode_stats=e.get('episode_stats'), gripper_kp=e.get('gripper_kp'), gripper_kd=e.get('gripper_kd'), gripper_force_limit=e.get('gripper_force_limit'), grip_close_sign=e.get('grip_close_sign'), pad_c0=PAD_C0, pad_gamma=SILICONE_GAMMA, pad_spec=PAD_SPEC)
    path = Path(e['log_dir']) / f"{e['env_tag']}_ep{e['episode']:04d}_traj.npz"
    e['buf'].dump(path, meta=meta)

def _flush_pending():
    while _PENDING:
        _write_entry(_PENDING.pop(0))

@atexit.register
def _mp_flush_all():
    _flush_pending()

class PhysConstraintMixin:

    def __init__(self, *args, maniphys_registry_path=None, maniphys_log_dir=None, maniphys_contact_every=5, **kwargs):
        self._mp_reg = PhysPropertyRegistry(maniphys_registry_path)
        log_dir = maniphys_log_dir or os.environ.get('MANIPHYS_LOG_DIR')
        self._mp_log_dir = Path(log_dir) if log_dir else None
        _ENV_COUNTER[0] += 1
        self._mp_env_tag = f'env{_ENV_COUNTER[0]:02d}'
        _flush_pending()
        self._mp_contact_every = max(1, int(maniphys_contact_every))
        self._mp_buf = None
        self._mp_ep_idx = -1
        self._mp_obj = None
        self._mp_pellets = []
        self._mp_pellet_scene = None
        self._mp_apply_model_db_override(kwargs)
        super().__init__(*args, **kwargs)

    def _mp_apply_model_db_override(self, kwargs):
        partial = self._mp_reg.model_db_override()
        for (mid, entry) in (kwargs.get('model_db_override') or {}).items():
            partial[mid] = {**partial.get(mid, {}), **entry}
        if not partial:
            return
        from mani_skill2_real2sim import format_path
        from mani_skill2_real2sim.utils.io_utils import load_json
        asset_root = Path(format_path(kwargs.get('asset_root') or type(self).DEFAULT_ASSET_ROOT))
        model_json = asset_root / format_path(kwargs.get('model_json') or type(self).DEFAULT_MODEL_JSON)
        base = load_json(model_json) if model_json.exists() else {}
        models_dir = asset_root / 'models'
        partial = {mid: e for (mid, e) in partial.items() if mid in base or (models_dir / mid).is_dir()}
        if not partial:
            return
        kwargs['model_db_override'] = {mid: {**base.get(mid, {}), **entry} for (mid, entry) in partial.items()}

    def _load_model(self):
        mid = getattr(self, 'model_id', None)
        if mid is not None:
            p = self._mp_reg.props(mid)
            self.obj_static_friction = p['static_friction']
            self.obj_dynamic_friction = p['dynamic_friction']
        super()._load_model()

    def _build_actor_helper(self, model_id, scene, scale=1.0, physical_material=None, density=1000.0, root_dir=None):
        from mani_skill2_real2sim import ASSET_DIR
        root = Path(root_dir) if root_dir is not None else ASSET_DIR / 'custom'
        ent = self._mp_reg.db.get(model_id, {})
        if 'static_friction' in ent or 'dynamic_friction' in ent:
            p = self._mp_reg.props(model_id)
            physical_material = scene.create_physical_material(static_friction=p['static_friction'], dynamic_friction=p['dynamic_friction'], restitution=p['restitution'])
        rm = ent.get('render_material')
        if rm is None:
            return super()._build_actor_helper(model_id, scene, scale=scale, physical_material=physical_material, density=density, root_dir=root)
        model_dir = root / 'models' / model_id
        builder = scene.create_actor_builder()
        builder.add_multiple_collisions_from_file(filename=str(model_dir / 'collision.obj'), scale=[scale] * 3, material=physical_material, density=density)
        visual_file = model_dir / 'textured.obj'
        if not visual_file.exists():
            visual_file = model_dir / 'textured.dae'
            if not visual_file.exists():
                visual_file = model_dir / 'textured.glb'
        mat = self._renderer.create_material()
        mat.set_base_color(rm['base_color'])
        mat.set_metallic(rm.get('metallic', 0.0))
        mat.set_roughness(rm.get('roughness', 0.4))
        mat.set_specular(rm.get('specular', 0.5))
        mat.set_transmission(rm.get('transmission', 0.0))
        builder.add_visual_from_file(filename=str(visual_file), scale=[scale] * 3, material=mat)
        return builder.build()

    def initialize_episode(self):
        self._mp_dump()
        ret = super().initialize_episode()
        self._mp_setup_episode()
        return ret

    def close(self):
        self._mp_dump()
        return super().close()

    def _mp_target_obj(self):
        return getattr(self, 'episode_source_obj', None) or getattr(self, 'obj', None)

    def _mp_model_id(self):
        mid = getattr(self, 'model_id', None)
        if mid:
            return mid
        obj = self._mp_target_obj()
        return obj.name if obj is not None else None

    def _mp_align_finger_friction(self):
        p = self._mp_props or {}
        (mu_s, mu_d) = (p.get('static_friction'), p.get('dynamic_friction'))
        if mu_s is None:
            self._mp_pad_mu = None
            return
        mat = self._scene.create_physical_material(float(mu_s), float(mu_d if mu_d is not None else mu_s), 0.0)
        n = 0
        for nm in ('finger_left_link', 'finger_right_link'):
            link = getattr(self.agent, nm, None)
            if link is None:
                continue
            for cs in link.get_collision_shapes():
                cs.set_physical_material(mat)
                n += 1
        self._mp_pad_mu = float(mu_s) if n else None

    def _mp_setup_episode(self):
        self._mp_ep_idx += 1
        self._mp_substep = 0
        self._mp_obj = obj = self._mp_target_obj()
        self._mp_buf = TrajBuffer() if self._mp_log_dir is not None else None
        self._mp_pending = None
        self._mp_contact = dict(pos_obj=np.full(3, np.nan), impulse=0.0)
        st = self._mp_state = dict(toppled=False, slipped=False, rot_slipped=False, spilled=False, first_violation={}, grasp_armed=False, grasped=False, grasp_true_count=0, grasp_false_count=0, rel_q0=None, phase='pre', tilt_exceed=0, settle_count=0, last_tilt=0.0)
        if obj is None:
            self._mp_props = None
            return
        self._mp_props = self._mp_reg.props(self._mp_model_id() or '')
        self._mp_align_finger_friction()
        self._mp_persist_substeps = int(self._mp_props.get('topple_persist_s', 0.5) * self._sim_freq)
        R0 = quat2mat(obj.pose.q)
        self._mp_up_local = R0.T @ np.array([0.0, 0.0, 1.0])
        self._mp_base_p = obj.pose.p.copy()
        st['baseline_z'] = float(obj.pose.p[2])
        self._mp_spawn_pellets(obj)
        if self._mp_buf is not None:
            gr = self._mp_gripper_controller()
            cfg = getattr(gr, 'config', None)
            self._mp_pending = dict(buf=self._mp_buf, state=st, props=self._mp_props, model_id=self._mp_model_id(), robot_uid=getattr(self, 'robot_uid', None), control_freq=self._control_freq, sim_freq=self._sim_freq, episode=self._mp_ep_idx, env_tag=self._mp_env_tag, log_dir=self._mp_log_dir, gripper_kp=float(np.mean(cfg.stiffness)) if cfg is not None else None, gripper_kd=float(np.mean(cfg.damping)) if cfg is not None else None, gripper_force_limit=float(np.mean(cfg.force_limit)) if cfg is not None else None, grip_close_sign=_close_sign(gr, cfg), success=None, episode_stats=None)
            _PENDING.append(self._mp_pending)

    def _mp_spawn_pellets(self, container):
        cfg = self._mp_props.get('spill') if self._mp_props else None
        if not cfg:
            self._mp_pellets = []
            return
        if self._mp_pellet_scene is not self._scene:
            self._mp_pellets = []
            self._mp_pellet_scene = self._scene
        (n, r) = (int(cfg.get('n_pellets', 15)), float(cfg.get('pellet_radius', 0.006)))
        color = list(cfg.get('color', [0.72, 0.55, 0.35]))
        if not self._mp_pellets:
            for i in range(n):
                b = self._scene.create_actor_builder()
                b.add_sphere_collision(radius=r, density=800)
                b.add_sphere_visual(radius=r, color=color)
                self._mp_pellets.append(b.build(name=f'mp_pellet_{i}'))
        base = container.pose.p
        spawn_z = float(cfg.get('spawn_z', 0.02))
        for (i, pel) in enumerate(self._mp_pellets):
            dx = (i % 3 - 1) * 2.1 * r
            dy = (i // 3 % 3 - 1) * 2.1 * r
            dz = spawn_z + i // 9 * 2.1 * r
            pel.set_pose(sapien.Pose([base[0] + dx, base[1] + dy, base[2] + dz]))
            pel.set_velocity([0, 0, 0])
            pel.set_angular_velocity([0, 0, 0])

    def _after_simulation_step(self):
        super()._after_simulation_step()
        obj = self._mp_obj
        if obj is None:
            return
        self._mp_substep += 1
        st = self._mp_state
        agent = self.agent
        self._mp_update_contact_detail(obj)
        if self._mp_substep % self._mp_contact_every == 0:
            raw_grasp = bool(agent.check_grasp(obj))
            if raw_grasp:
                st['grasp_true_count'] += 1
                st['grasp_false_count'] = 0
            else:
                st['grasp_false_count'] += 1
                st['grasp_true_count'] = 0
            if not st['grasped'] and st['grasp_true_count'] >= _GRASP_DEBOUNCE:
                st['grasped'] = True
                st['grasp_armed'] = True
                st['phase'] = 'grasped'
                tcp_q = self.tcp.pose.q
                st['rel_q0'] = qmult(qinverse(tcp_q), obj.pose.q)
            elif st['grasped'] and st['grasp_false_count'] >= _GRASP_DEBOUNCE:
                st['grasped'] = False
                st['phase'] = 'released'
                st['settle_count'] = 0
                if not st['slipped'] and self._mp_gripper_closed_cmd():
                    self._mp_flag('slipped')
        tilt_deg = self._mp_tilt_deg(obj)
        st['last_tilt'] = float(tilt_deg)
        if self._mp_props.get('standing'):
            if st['phase'] == 'pre' and (not st['toppled']):
                if tilt_deg > self._mp_props['tilt_limit_deg']:
                    st['tilt_exceed'] += 1
                    if st['tilt_exceed'] >= self._mp_persist_substeps:
                        self._mp_flag('toppled')
                else:
                    st['tilt_exceed'] = 0
            elif st['phase'] == 'released':
                v = float(np.linalg.norm(obj.get_velocity()))
                w = float(np.linalg.norm(obj.get_angular_velocity()))
                if v < 0.02 and w < 0.2:
                    st['settle_count'] += 1
                    if st['settle_count'] >= 50:
                        R0 = quat2mat(obj.pose.q)
                        self._mp_up_local = R0.T @ np.array([0.0, 0.0, 1.0])
                        st['baseline_z'] = float(obj.pose.p[2])
                        st['phase'] = 'pre'
                        st['tilt_exceed'] = 0
                else:
                    st['settle_count'] = 0
        if st['grasped'] and st['rel_q0'] is not None and (not st['rot_slipped']):
            rel_q = qmult(qinverse(self.tcp.pose.q), obj.pose.q)
            dq = qmult(rel_q, qinverse(st['rel_q0']))
            drift_deg = np.degrees(2.0 * np.arccos(np.clip(abs(dq[0]), -1.0, 1.0)))
            if drift_deg > self._mp_props['rot_slip_limit_deg']:
                self._mp_flag('rot_slipped')
        if self._mp_pellets and (not st['spilled']):
            if self._mp_check_spill(obj):
                self._mp_flag('spilled')
        if self._mp_buf is not None:
            gr = self._mp_gripper_controller()
            gcmd = np.asarray(getattr(gr, '_last_drive_qpos_targets', [np.nan, np.nan]), dtype=np.float64) if gr is not None else np.array([np.nan, np.nan])
            gq = gr.qpos if gr is not None else np.array([np.nan, np.nan])
            gv = gr.qvel if gr is not None else np.array([np.nan, np.nan])
            self._mp_buf.append(control_step=int(self._elapsed_steps), substep=int(self._mp_substep), obj_p=obj.pose.p, obj_q=obj.pose.q, obj_v=obj.get_velocity(), obj_w=obj.get_angular_velocity(), tcp_p=self.tcp.pose.p, tcp_q=self.tcp.pose.q, tcp_v=self.tcp.get_velocity(), grip_qpos=gq, grip_qvel=gv, grip_cmd=gcmd, pd_force=self._mp_pd_force(gr, gq, gv, gcmd), contact_pos_obj=self._mp_contact['pos_obj'], contact_impulse=float(self._mp_contact['impulse']), contact_time_s=float(self._mp_substep / self._sim_freq), physics_dt=float(1.0 / self._sim_freq), contact_normal_dyad_L=self._mp_contact['normal_dyad_L'], contact_normal_dyad_R=self._mp_contact['normal_dyad_R'], contact_normal_force_L=self._mp_contact['normal_force_L'], contact_normal_force_R=self._mp_contact['normal_force_R'], contact_pos_L=self._mp_contact.get('pos_L', np.full(3, np.nan)), contact_pos_R=self._mp_contact.get('pos_R', np.full(3, np.nan)), contact_f_L=float(self._mp_contact.get('f_L', 0.0)), contact_f_R=float(self._mp_contact.get('f_R', 0.0)), grasped=int(st['grasped']), tilt_deg=float(tilt_deg), flags=np.array([st['toppled'], st['slipped'], st['rot_slipped'], st['spilled']], dtype=np.int8))

    def _mp_update_contact_detail(self, obj):
        named = [(k, getattr(self.agent, n, None)) for (k, n) in (('L', 'finger_left_link'), ('R', 'finger_right_link'))]
        named = [(k, f) for (k, f) in named if f is not None]
        fingers = [f for (_, f) in named]
        if not fingers:
            self._mp_contact = dict(pos_obj=np.full(3, np.nan), impulse=0.0)
            return
        contacts = self._scene.get_contacts()
        (points, imp_total) = ([], 0.0)
        side = {'L': [], 'R': []}
        for (k, f) in named:
            for (contact, flag) in get_pairwise_contacts(contacts, f, obj):
                for pt in contact.points:
                    imp = float(np.linalg.norm(pt.impulse))
                    if imp > 0:
                        points.append(pt.position)
                        imp_total += imp
                        side[k].append((np.asarray(pt.position, float), imp, np.asarray(pt.normal, float) * (-1 if flag else 1)))
        if points:
            inv = obj.pose.inv()
            local = [inv.transform(sapien.Pose(p)).p for p in points]
            self._mp_contact = dict(pos_obj=np.mean(local, axis=0), impulse=imp_total, **self._mp_sides(side, inv, self._sim_freq))
        else:
            self._mp_contact = dict(pos_obj=np.full(3, np.nan), impulse=0.0, **self._mp_sides({'L': [], 'R': []}, None, None))

    @staticmethod
    def _mp_sides(side, inv, sim_freq):
        out = {}
        for (k, v) in side.items():
            if v:
                P = np.array([p for (p, _, _) in v])
                w = np.array([i for (_, i, _) in v])
                normals = np.array([normal for (_, _, normal) in v])
                R = quat2mat(inv.q)
                normals = normals @ R.T
                out[f'normal_dyad_{k}'] = np.einsum('n,ni,nj->ij', w, normals, normals) / w.sum()
                out[f'normal_force_{k}'] = (normals * w[:, None]).sum(0) * float(sim_freq)
                ctr = (P * w[:, None]).sum(0) / w.sum()
                out[f'pos_{k}'] = inv.transform(sapien.Pose(ctr)).p
                out[f'f_{k}'] = float(w.sum()) * float(sim_freq)
            else:
                out[f'pos_{k}'] = np.full(3, np.nan)
                out[f'f_{k}'] = 0.0
                out[f'normal_dyad_{k}'] = np.zeros((3, 3))
                out[f'normal_force_{k}'] = np.zeros(3)
        return out

    @staticmethod
    def _mp_pd_force(gr, qpos, qvel, target):
        cfg = getattr(gr, 'config', None)
        if cfg is None or np.any(np.isnan(target)):
            return np.array([np.nan, np.nan])
        f = cfg.stiffness * (target - qpos) - cfg.damping * qvel
        return np.clip(f, -cfg.force_limit, cfg.force_limit)

    def _mp_flag(self, axis):
        st = self._mp_state
        st[axis] = True
        st['first_violation'][axis] = (int(self._elapsed_steps), int(self._mp_substep))

    def _mp_tilt_deg(self, obj):
        up_world = quat2mat(obj.pose.q) @ self._mp_up_local
        return float(np.degrees(np.arccos(np.clip(up_world[2], -1.0, 1.0))))

    def _mp_gripper_controller(self):
        ctrl = getattr(self.agent, 'controller', None)
        subs = getattr(ctrl, 'controllers', None)
        return subs.get('gripper') if isinstance(subs, dict) else None

    def _mp_gripper_closed_cmd(self):
        gr = self._mp_gripper_controller()
        tgt = getattr(gr, '_last_drive_qpos_targets', None) if gr is not None else None
        if tgt is None:
            fn = getattr(self.agent, 'get_gripper_closedness', None)
            return bool(fn() > 0.7) if fn is not None else False
        return bool(np.mean(tgt) < 0.026)

    def _mp_check_spill(self, container):
        cfg = self._mp_props.get('spill') or {}
        rim_z = float(cfg.get('rim_z', 0.05))
        rim_r = float(cfg.get('rim_radius', 0.04))
        inv = container.pose.inv()
        for pel in self._mp_pellets:
            lp = inv.transform(pel.pose).p
            if np.hypot(lp[0], lp[1]) > rim_r * 1.15 and lp[2] < rim_z:
                return True
        return False
    _MP_GLASS_BLEND = 0.55

    def get_obs(self, *args, **kwargs):
        obs = super().get_obs(*args, **kwargs)
        if self._obs_mode != 'image' or getattr(self, 'rgb_overlay_img', None) is None:
            return obs
        glass_names = {mid for (mid, e) in self._mp_reg.db.items() if e.get('render_material', {}).get('base_color', [1, 1, 1, 1])[3] < 1.0}
        if not glass_names:
            return obs
        glass_ids = [a.id for a in self.get_actors() if a.name in glass_names]
        if not glass_ids:
            return obs
        import cv2
        for cam in self.rgb_overlay_cameras:
            if cam not in obs['image'] or 'Segmentation' not in obs['image'][cam]:
                continue
            m = np.isin(obs['image'][cam]['Segmentation'][..., 1], glass_ids)
            if not m.any():
                continue
            color = obs['image'][cam]['Color']
            ov = cv2.resize(self.rgb_overlay_img, (color.shape[1], color.shape[0]))
            color[..., :3][m] = color[..., :3][m] * self._MP_GLASS_BLEND + ov[m] * (1.0 - self._MP_GLASS_BLEND)
        return obs

    def evaluate(self, **kwargs):
        info = super().evaluate(**kwargs)
        if self._mp_obj is None:
            return info
        st = self._mp_state
        viol = dict(toppled=st['toppled'], slipped=st['slipped'], rot_slipped=st['rot_slipped'], spilled=st['spilled'])
        info['mp_violation'] = any(viol.values())
        for (k, v) in viol.items():
            info[f'mp_{k}'] = v
        es = getattr(self, 'episode_stats', None)
        if isinstance(es, (dict, OrderedDict)):
            es['mp_violation'] = bool(info['mp_violation'])
        p = getattr(self, '_mp_pending', None)
        if p is not None:
            p['success'] = bool(info.get('success', False))
            if isinstance(es, (dict, OrderedDict)):
                p['episode_stats'] = {k: bool(v) if isinstance(v, (bool, np.bool_)) else v for (k, v) in es.items()}
        return info

    def _mp_dump(self):
        p = getattr(self, '_mp_pending', None)
        if p is not None and p in _PENDING:
            _PENDING.remove(p)
            _write_entry(p)
        self._mp_pending = None
        self._mp_buf = None

    def __del__(self):
        self._mp_dump()

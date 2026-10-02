import os
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
from transforms3d.quaternions import quat2mat, qmult, qinverse
from maniphysics.bench.logger import TrajBuffer
from maniphysics.bench.properties import PhysPropertyRegistry
_GRASP_DEBOUNCE = 3
_ENV_COUNTER = [0]
FRICTION_MODES = ('pair', 'geom', 'off')
_PAIR_TAG = 'maniphys_padpair'

class ManiPhysMujocoMixin:
    PAD_C0 = 0.000854
    PAD_GAMMA = 0.307
    PAD_SPEC = 'Silicone pad: E=20 MPa, thickness=2 mm'

    def __init__(self, *args, maniphys_model_id=None, maniphys_obj_name=None, maniphys_log_dir=None, maniphys_registry_path=None, maniphys_contact_every=5, maniphys_pad=None, maniphys_finger_friction=None, **kwargs):
        self._mp_reg = PhysPropertyRegistry(maniphys_registry_path)
        self._mp_model_id_req = maniphys_model_id
        self._mp_obj_name_req = maniphys_obj_name
        fmode = (maniphys_finger_friction or os.environ.get('MANIPHYS_FINGER_FRICTION') or 'pair').strip().lower()
        self._mp_fmode = fmode if fmode in FRICTION_MODES else 'pair'
        self._mp_registered = bool(maniphys_model_id and maniphys_model_id in self._mp_reg.db)
        self._mp_pairs = []
        self._mp_pad_mu = None
        log_dir = maniphys_log_dir or os.environ.get('MANIPHYS_LOG_DIR')
        self._mp_log_dir = Path(log_dir) if log_dir else None
        _ENV_COUNTER[0] += 1
        self._mp_env_tag = f'env{_ENV_COUNTER[0]:02d}'
        self._mp_contact_every = max(1, int(maniphys_contact_every))
        pad = maniphys_pad or {}
        self._mp_pad = dict(c0=pad.get('c0', self.PAD_C0), gamma=pad.get('gamma', self.PAD_GAMMA), spec=pad.get('spec', self.PAD_SPEC))
        self._mp_buf = None
        self._mp_ep_idx = -1
        self._mp_ready = False
        super().__init__(*args, **kwargs)

    def _initialize_sim(self, xml_string=None):
        xml = xml_string if xml_string is not None else self.model.get_xml()
        if getattr(self, '_mp_fmode', 'off') == 'pair' and getattr(self, '_mp_registered', False):
            xml = self._mp_inject_friction_pairs(xml)
        return super()._initialize_sim(xml_string=xml)

    def _mp_inject_friction_pairs(self, xml):
        self._mp_pairs = []
        try:
            root = ET.fromstring(xml)
            if root.find(f".//pair[@name='{_PAIR_TAG}0']") is not None:
                return xml
            robot = self.robots[0]
            pads = []
            for k in ('left_finger', 'right_finger', 'left_fingerpad', 'right_fingerpad'):
                pads += list(robot.gripper.important_geoms.get(k, []))
            pads = sorted(set(pads))
            objs = self._mp_xml_obj_geoms(root)
            if not pads or not objs:
                return xml
            contact = root.find('contact')
            if contact is None:
                contact = ET.SubElement(root, 'contact')
            for (i, (a, b)) in enumerate(((p, o) for p in pads for o in objs)):
                ET.SubElement(contact, 'pair', {'name': f'{_PAIR_TAG}{i}', 'geom1': a, 'geom2': b})
                self._mp_pairs.append((a, b))
            return ET.tostring(root, encoding='unicode')
        except Exception as exc:
            self._mp_pair_error = f'{type(exc).__name__}: {exc}'
            self._mp_pairs = []
            return xml

    def _mp_xml_obj_geoms(self, root):
        name = self._mp_obj_name_req
        if name is None:
            objs = getattr(self, 'objects_dict', None) or {}
            name = list(objs.keys())[0] if objs else None
        if name is None:
            return []
        body = None
        for cand in (f'{name}_main', name, f'{name}_root'):
            body = root.find(f".//body[@name='{cand}']")
            if body is not None:
                break
        if body is None:
            return []
        out = []
        for g in body.iter('geom'):
            gn = g.get('name')
            if not gn:
                continue
            if g.get('contype', '1') == '0' and g.get('conaffinity', '1') == '0':
                continue
            out.append(gn)
        return out

    def _mp_apply_friction(self, sim):
        m = sim.model
        mu = self._mp_props.get('static_friction')
        info = dict(mode=self._mp_fmode, obj_mu=None if mu is None else float(mu), n_pairs=0, pad_mu=None)
        if not self._mp_registered:
            return dict(mode='off', obj_mu=None, n_pairs=0, pad_mu=None)
        if getattr(self, '_mp_pair_error', None):
            info['pair_error'] = self._mp_pair_error
        if self._mp_fmode == 'pair':
            done = 0
            for p in range(int(m.npair)):
                (g1, g2) = (int(m.pair_geom1[p]), int(m.pair_geom2[p]))
                pair = {g1, g2}
                if not (pair & self._mp_obj_geoms and pair & self._mp_pad_geoms):
                    continue
                go = g1 if g1 in self._mp_obj_geoms else g2
                gp = g2 if go == g1 else g1
                fo = np.asarray(m.geom_friction[go], dtype=float)
                m.pair_friction[p] = [fo[0], fo[0], fo[1], fo[2], fo[2]]
                m.pair_dim[p] = int(max(m.geom_condim[go], m.geom_condim[gp]))
                (s_o, s_p) = (float(m.geom_solmix[go]), float(m.geom_solmix[gp]))
                mix = 0.5 if s_o < 1e-15 and s_p < 1e-15 else 0.0 if s_o < 1e-15 else 1.0 if s_p < 1e-15 else s_o / (s_o + s_p)
                (r_o, r_p) = (np.asarray(m.geom_solref[go]), np.asarray(m.geom_solref[gp]))
                m.pair_solref[p] = mix * r_o + (1 - mix) * r_p if r_o[0] > 0 and r_p[0] > 0 else np.minimum(r_o, r_p)
                (i_o, i_p) = (np.asarray(m.geom_solimp[go]), np.asarray(m.geom_solimp[gp]))
                m.pair_solimp[p] = mix * i_o + (1 - mix) * i_p
                m.pair_margin[p] = max(float(m.geom_margin[go]), float(m.geom_margin[gp]))
                m.pair_gap[p] = max(float(m.geom_gap[go]), float(m.geom_gap[gp]))
                done += 1
            info['n_pairs'] = done
            if done:
                return info
            raise RuntimeError('Contact pair injection failed')
        if self._mp_fmode != 'off' and mu is not None:
            for g in self._mp_pad_geoms:
                m.geom_friction[g][0] = float(mu)
            info['pad_mu'] = float(mu)
        return info

    def _mp_friction_info(self):
        info = dict(getattr(self, '_mp_frict', {'mode': getattr(self, '_mp_fmode', '?')}))
        d = getattr(self, '_mp_d', None)
        if d is not None and getattr(self, '_mp_ready', False):
            mus = []
            for i in range(d.ncon):
                c = d.contact[i]
                (g1, g2) = (int(c.geom1), int(c.geom2))
                if g1 in self._mp_pad_geoms and g2 in self._mp_obj_geoms or (g2 in self._mp_pad_geoms and g1 in self._mp_obj_geoms):
                    mus.append(round(float(c.friction[0]), 6))
            if mus:
                info['effective_mu_now'] = sorted(set(mus))
        return info

    def _mp_setup(self):
        (sim, robot) = (self.sim, self.robots[0])
        m = sim.model._model
        (self._mp_m, self._mp_d) = (m, sim.data._data)
        objs = getattr(self, 'objects_dict', None) or {}
        name = self._mp_obj_name_req or (list(objs.keys())[0] if objs else None)
        self._mp_obj_name = name
        self._mp_body_id = None
        if name is not None:
            for cand in (f'{name}_main', name, f'{name}_root'):
                try:
                    self._mp_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cand)
                    if self._mp_body_id >= 0:
                        break
                except Exception:
                    pass
        if self._mp_body_id is None or self._mp_body_id < 0:
            self._mp_ready = False
            return
        self._mp_obj_geoms = {g for g in range(m.ngeom) if self._mp_subtree_has(m, m.geom_bodyid[g], self._mp_body_id)}
        pad_names = []
        (self._mp_pad_L, self._mp_pad_R) = (set(), set())
        for k in ('left_finger', 'right_finger', 'left_fingerpad', 'right_fingerpad'):
            names = robot.gripper.important_geoms.get(k, [])
            pad_names += names
            side = self._mp_pad_L if k.startswith('left') else self._mp_pad_R
            for n in names:
                gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
                if gid >= 0:
                    side.add(gid)
        self._mp_pad_geoms = set()
        for n in pad_names:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
            if gid >= 0:
                self._mp_pad_geoms.add(gid)
        self._mp_obj_geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in self._mp_obj_geoms]
        self._mp_qpos_ids = list(robot._ref_gripper_joint_pos_indexes)
        self._mp_qvel_ids = list(getattr(robot, '_ref_gripper_joint_vel_indexes', self._mp_qpos_ids))
        self._mp_act_ids = list(robot._ref_joint_gripper_actuator_indexes)
        self._mp_kp = float(sim.model.actuator_gainprm[self._mp_act_ids[0]][0])
        self._mp_close_sign = []
        for a in self._mp_act_ids:
            cr = np.array(sim.model.actuator_ctrlrange[a], dtype=float)
            self._mp_close_sign.append(1.0 if abs(cr[0]) < abs(cr[1]) else -1.0)
        fr = sim.model.actuator_forcerange[self._mp_act_ids[0]]
        self._mp_flim = float(abs(fr[1])) if fr[1] != 0 else 1000000000.0
        self._mp_eef_site = robot.eef_site_id
        self._mp_model_id = self._mp_model_id_req or name
        self._mp_props = self._mp_reg.props(self._mp_model_id)
        self._mp_frict = self._mp_apply_friction(sim)
        self._mp_pad_mu = self._mp_frict.get('pad_mu')
        self._mp_pellet_bodies = []
        for (nm, o) in (objs or {}).items():
            if 'pellet' not in nm:
                continue
            for cand in (f'{nm}_main', nm):
                pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cand)
                if pid >= 0:
                    self._mp_pellet_bodies.append(pid)
                    break
        bb = self._mp_props.get('bbox') or {}
        self._mp_rim = float(max(abs(bb.get('max', [0.08])[0]), abs(bb.get('min', [-0.08])[0])))
        self._mp_rim_h = float(bb.get('max', [0, 0, 0.05])[2]) if bb.get('max') else 0.05
        R0 = quat2mat(np.array(sim.data.body_xquat[self._mp_body_id], dtype=np.float64))
        self._mp_up_local = R0.T @ np.array([0.0, 0.0, 1.0])
        self._mp_state = dict(toppled=False, slipped=False, rot_slipped=False, spilled=False, grasped=False, grasp_true_count=0, grasp_false_count=0, grasp_armed=False, phase='pre', settle_count=0, rel_q0=None, tilt_exceed=0, last_tilt=0.0, first_violation={})
        self._mp_contact = dict(pos_obj=np.full(3, np.nan), impulse=0.0)
        self._mp_substep = 0
        self._mp_ready = True

    @staticmethod
    def _mp_subtree_has(m, body_id, root_id):
        b = body_id
        while b > 0:
            if b == root_id:
                return True
            b = m.body_parentid[b]
        return body_id == root_id

    def reset(self, *args, **kwargs):
        self._mp_dump()
        out = super().reset(*args, **kwargs)
        self._mp_setup()
        self._mp_place_pellets()
        self._mp_ep_idx += 1
        self._mp_buf = TrajBuffer() if self._mp_log_dir and self._mp_ready else None
        return out
    _mp_slip_drop_min = 0.03

    def _mp_drop_height(self, obj_p):
        bb = self._mp_props.get('bbox') or {}
        half = abs(float(bb.get('min', [0, 0, -0.02])[2])) if bb.get('min') else 0.02
        bottom = float(obj_p[2]) - half
        best = None
        for gid in range(self._mp_m.ngeom):
            if gid in self._mp_obj_geoms or gid in self._mp_pad_geoms:
                continue
            gp = self._mp_d.geom_xpos[gid]
            if np.hypot(gp[0] - obj_p[0], gp[1] - obj_p[1]) > 0.15:
                continue
            top = float(gp[2]) + float(np.max(self._mp_m.geom_size[gid]))
            if top <= bottom + 0.0001:
                best = top if best is None else max(best, top)
        return None if best is None else bottom - best

    def _mp_place_pellets(self, ring_frac=0.45, drop_h=0.02):
        if not getattr(self, '_mp_ready', False) or not self._mp_pellet_bodies:
            return
        (sim, m) = (self.sim, self._mp_m)
        cpos = np.array(sim.data.body_xpos[self._mp_body_id], dtype=np.float64)
        r = self._mp_rim * ring_frac
        n = len(self._mp_pellet_bodies)
        for (i, pid) in enumerate(self._mp_pellet_bodies):
            jadr = m.body_jntadr[pid]
            if jadr < 0:
                continue
            qadr = m.jnt_qposadr[jadr]
            th = 2.0 * np.pi * i / max(n, 1)
            sim.data.qpos[qadr:qadr + 3] = [cpos[0] + r * np.cos(th), cpos[1] + r * np.sin(th), cpos[2] + drop_h]
            sim.data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
            dadr = m.jnt_dofadr[jadr]
            sim.data.qvel[dadr:dadr + 6] = 0.0
        sim.forward()
        for _ in range(60):
            sim.step()
        sim.forward()

    def _update_observables(self, force=False):
        super()._update_observables(force=force)
        if not getattr(self, '_mp_ready', False):
            return
        self._mp_substep += 1
        (sim, st) = (self.sim, self._mp_state)
        (m, d) = (self._mp_m, self._mp_d)
        bid = self._mp_body_id
        obj_p = np.array(sim.data.body_xpos[bid], dtype=np.float64)
        obj_q = np.array(sim.data.body_xquat[bid], dtype=np.float64)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
        (obj_w, obj_v) = (vel[:3].copy(), vel[3:].copy())
        self._mp_update_contact(obj_p, obj_q)
        if self._mp_substep % self._mp_contact_every == 0:
            raw_grasp = bool(self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self._mp_obj_geom_names))
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
                st['rel_q0'] = qmult(qinverse(self._mp_tcp_quat()), obj_q)
            elif st['grasped'] and st['grasp_false_count'] >= _GRASP_DEBOUNCE:
                st['grasped'] = False
                st['phase'] = 'released'
                st['settle_count'] = 0
                if not st['slipped'] and self._mp_gripper_closing():
                    drop = self._mp_drop_height(obj_p)
                    if drop is None or drop > self._mp_slip_drop_min:
                        self._mp_flag('slipped')
        tilt_deg = float(np.degrees(np.arccos(np.clip((quat2mat(obj_q) @ self._mp_up_local)[2], -1.0, 1.0))))
        st['last_tilt'] = tilt_deg
        if self._mp_props.get('standing') and st['phase'] == 'pre' and (not st['toppled']):
            if tilt_deg > self._mp_props['tilt_limit_deg']:
                st['tilt_exceed'] += 1
                if st['tilt_exceed'] >= 25:
                    self._mp_flag('toppled')
            else:
                st['tilt_exceed'] = 0
        if self._mp_pellet_bodies and (not st['spilled']):
            R = quat2mat(obj_q)
            for pid in self._mp_pellet_bodies:
                loc = R.T @ (np.array(sim.data.body_xpos[pid], dtype=np.float64) - obj_p)
                if np.hypot(loc[0], loc[1]) > self._mp_rim * 1.15 or loc[2] < -self._mp_rim_h:
                    self._mp_flag('spilled')
                    break
        if st['grasped'] and st['rel_q0'] is not None and (not st['rot_slipped']):
            dq = qmult(qmult(qinverse(self._mp_tcp_quat()), obj_q), qinverse(st['rel_q0']))
            drift = np.degrees(2.0 * np.arccos(np.clip(abs(dq[0]), -1.0, 1.0)))
            if drift > self._mp_props['rot_slip_limit_deg']:
                self._mp_flag('rot_slipped')
        if self._mp_buf is not None:
            gq = np.array(sim.data.qpos[self._mp_qpos_ids], dtype=np.float64)
            gv = np.array(sim.data.qvel[self._mp_qvel_ids], dtype=np.float64)
            gcmd = np.array(sim.data.ctrl[self._mp_act_ids], dtype=np.float64)
            pdf = np.array(sim.data.actuator_force[self._mp_act_ids], dtype=np.float64)
            self._mp_buf.append(control_step=int(getattr(self, 'timestep', 0)), substep=int(self._mp_substep), obj_p=obj_p, obj_q=obj_q, obj_v=obj_v, obj_w=obj_w, tcp_p=np.array(sim.data.site_xpos[self._mp_eef_site], dtype=np.float64), tcp_q=self._mp_tcp_quat(), tcp_v=self._mp_site_vel(), grip_qpos=gq, grip_qvel=gv, grip_cmd=gcmd, pd_force=pdf, contact_pos_obj=self._mp_contact['pos_obj'], contact_impulse=float(self._mp_contact['impulse']), contact_time_s=float(self._mp_substep * m.opt.timestep), physics_dt=float(m.opt.timestep), contact_normal_dyad_L=self._mp_contact['normal_dyad_L'], contact_normal_dyad_R=self._mp_contact['normal_dyad_R'], contact_normal_force_L=self._mp_contact['normal_force_L'], contact_normal_force_R=self._mp_contact['normal_force_R'], contact_pos_L=self._mp_contact.get('pos_L', np.full(3, np.nan)), contact_pos_R=self._mp_contact.get('pos_R', np.full(3, np.nan)), contact_f_L=float(self._mp_contact.get('f_L', 0.0)), contact_f_R=float(self._mp_contact.get('f_R', 0.0)), grasped=int(st['grasped']), tilt_deg=tilt_deg, flags=np.array([st['toppled'], st['slipped'], st['rot_slipped'], st['spilled']], dtype=np.int8))

    def _mp_tcp_quat(self):
        import robosuite.utils.transform_utils as T
        xmat = np.array(self.sim.data.site_xmat[self._mp_eef_site]).reshape(3, 3)
        q_xyzw = T.mat2quat(xmat)
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    def _mp_site_vel(self):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self._mp_m, self._mp_d, mujoco.mjtObj.mjOBJ_SITE, self._mp_eef_site, v, 0)
        return v[3:].copy()

    def _mp_gripper_closing(self):
        cmd = np.array(self.sim.data.ctrl[self._mp_act_ids], dtype=np.float64)
        return bool(cmd[0] < 0.02)

    def _mp_update_contact(self, obj_p, obj_q):
        (m, d) = (self._mp_m, self._mp_d)
        (pts, imp) = ([], 0.0)
        side_pts = {'L': [], 'R': []}
        buf = np.zeros(6, dtype=np.float64)
        dt = float(m.opt.timestep)
        for i in range(d.ncon):
            c = d.contact[i]
            (g1, g2) = (int(c.geom1), int(c.geom2))
            pad_obj = g1 in self._mp_pad_geoms and g2 in self._mp_obj_geoms or (g2 in self._mp_pad_geoms and g1 in self._mp_obj_geoms)
            if not pad_obj:
                continue
            mujoco.mj_contactForce(m, d, i, buf)
            f = float(abs(buf[0]))
            if f > 0:
                pts.append(np.array(c.pos, dtype=np.float64))
                imp += f * dt
                if g1 in self._mp_pad_L or g2 in self._mp_pad_L:
                    side_pts['L'].append((np.array(c.pos, dtype=np.float64), f, np.asarray(c.frame[:3], dtype=np.float64) * (1 if g2 in self._mp_obj_geoms else -1)))
                elif g1 in self._mp_pad_R or g2 in self._mp_pad_R:
                    side_pts['R'].append((np.array(c.pos, dtype=np.float64), f, np.asarray(c.frame[:3], dtype=np.float64) * (1 if g2 in self._mp_obj_geoms else -1)))
        if pts:
            R = quat2mat(obj_q)
            local = [R.T @ (p - obj_p) for p in pts]
            self._mp_contact = dict(pos_obj=np.mean(local, axis=0), impulse=imp, **self._mp_sides(side_pts, R, obj_p))
        else:
            self._mp_contact = dict(pos_obj=np.full(3, np.nan), impulse=0.0, **self._mp_sides({'L': [], 'R': []}, None, None))

    @staticmethod
    def _mp_sides(side_pts, R, obj_p):
        out = {}
        for (k, v) in side_pts.items():
            if v:
                P = np.array([p for (p, _, _) in v])
                w = np.array([f for (_, f, _) in v])
                normals = np.array([normal for (_, _, normal) in v]) @ R
                out[f'normal_dyad_{k}'] = np.einsum('n,ni,nj->ij', w, normals, normals) / w.sum()
                out[f'normal_force_{k}'] = (normals * w[:, None]).sum(0)
                ctr = (P * w[:, None]).sum(0) / w.sum()
                out[f'pos_{k}'] = R.T @ (ctr - obj_p)
                out[f'f_{k}'] = float(w.sum())
            else:
                out[f'pos_{k}'] = np.full(3, np.nan)
                out[f'f_{k}'] = 0.0
                out[f'normal_dyad_{k}'] = np.zeros((3, 3))
                out[f'normal_force_{k}'] = np.zeros(3)
        return out

    def _mp_flag(self, axis):
        st = self._mp_state
        st[axis] = True
        st['first_violation'][axis] = (int(getattr(self, 'timestep', 0)), int(self._mp_substep))

    def _mp_dump(self, success=None):
        buf = getattr(self, '_mp_buf', None)
        if buf is None or len(buf) == 0 or (not self._mp_log_dir):
            return
        st = self._mp_state
        meta = dict(episode=self._mp_ep_idx, model_id=self._mp_model_id, robot_uid='panda', control_freq=float(getattr(self, 'control_freq', 20)), sim_freq=float(1.0 / self._mp_m.opt.timestep), physics_dt=float(self._mp_m.opt.timestep), contact_every=1, contact_schema='normal-dyad', props=self._mp_props, violations={k: st[k] for k in ('toppled', 'slipped', 'rot_slipped', 'spilled')}, first_violation=st['first_violation'], final_tilt_deg=st.get('last_tilt'), success=success, obj_mass=float(self._mp_m.body_mass[self._mp_body_id]), obj_inertia=[float(x) for x in self._mp_m.body_inertia[self._mp_body_id]], obj_iquat=[float(x) for x in self._mp_m.body_iquat[self._mp_body_id]], obj_ipos=[float(x) for x in self._mp_m.body_ipos[self._mp_body_id]], gripper_kp=self._mp_kp, gripper_kd=0.0, gripper_force_limit=self._mp_flim, gripper_class=type(self.robots[0].gripper).__name__, platform='libero-robosuite-mujoco', grip_close_sign=list(self._mp_close_sign), finger_friction=self._mp_friction_info(), pad_c0=self._mp_pad['c0'], pad_gamma=self._mp_pad['gamma'], pad_spec=self._mp_pad['spec'])
        self._mp_log_dir.mkdir(parents=True, exist_ok=True)
        path = self._mp_log_dir / f'{self._mp_env_tag}_ep{self._mp_ep_idx:04d}_traj.npz'
        buf.dump(path, meta=meta)
        self._mp_buf = None

    def close(self):
        self._mp_dump()
        return super().close()

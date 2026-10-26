"""Dobot Magician (real community CAD model) in PyBullet: red box -> left stack, blue box -> right.

Robot: dobot_magician.urdf (DAE meshes from the widely-distributed Dobot Magician
community model; joint limits widened to official specs; per-link materials added).
Kinematics: warm-started numerical IK (PyBullet LAPACK) on the serial chain with a
fixed tool-down orientation, TCP = magician_link_5 origin + (0, 0, -0.06).

State:  [x, y, z, r, suction]   (TCP in base frame; r kept 0)
Action: [dx, dy, dz, dr, suction_target]
Camera: single oblique overhead RGB via getCameraImage (hardware GL + shadows when available).
"""
import os

import numpy as np
import pybullet as p
import pybullet_data

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "magician")
URDF_PATH = os.path.join(MODEL_DIR, "dobot_magician.urdf")
CUP_LOCAL = np.array([0.0, 0.0, -0.064])  # visual cup bottom in link5 frame (stereo-triangulated)
CUP_DROP = 0.0

TABLE_Z = 0.0


def _qmul(a, b):
    """Hamilton product of quaternions (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz])

HOME_XYZR = (0.20, 0.0, 0.13, 0.0)
PICK_ZONE = {"x": (0.14, 0.22), "y": (-0.08, 0.08)}
STACKS = {"left": {"x": 0.13, "y": -0.15}, "right": {"x": 0.13, "y": 0.15}}
BOX_SIZE = 0.04

COLORS = {
    "red": [0.85, 0.05, 0.05, 1],
    "blue": [0.05, 0.1, 0.85, 1],
}


class DobotStackEnv:
    def __init__(self, gui=False, seed=None, render_size=256):
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1 / 240)
        self.rng = np.random.default_rng(seed)
        self.render_size = render_size
        self.cam_dist, self.cam_pitch, self.cam_yaw = 0.55, -60, 30
        self._load_scene()

    def _load_scene(self):
        p.resetSimulation()
        self.box = None
        self.carrying = False
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.arm = p.loadURDF(URDF_PATH, useFixedBase=True)
        n = p.getNumJoints(self.arm)
        self.movable = [i for i in range(n)
                        if p.getJointInfo(self.arm, i)[2] in (0, 3)]
        self.link_idx = next(i for i in range(n) if p.getJointInfo(self.arm, i)[1].decode()
                             == "magician_joint_5")
        self.lo = [p.getJointInfo(self.arm, j)[8] for j in self.movable]
        self.hi = [p.getJointInfo(self.arm, j)[9] for j in self.movable]
        self._warm = [0.3] * len(self.movable)
        self._ik_target_orn = [0, 0, 0, 1]   # tool straight down (model home pose)
        # make the arm effectively kinematic: near-zero masses + heavy damping so
        # gravity cannot sag it between control ticks (boxes stay physical)
        for l in range(p.getNumJoints(self.arm)):
            p.changeDynamics(self.arm, l, mass=1e-4, linearDamping=50.0, angularDamping=50.0)
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.cid)
        except Exception:
            pass
        self._move_to(*HOME_XYZR[:3], iters=80)
        self.box = None
        self.carrying = False

    # ---------- kinematics (own analytic FK of the URDF serial chain) ----------
    @staticmethod
    def _ry(t, v):
        c, s = np.cos(t), np.sin(t)
        return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]])

    @classmethod
    def _fk(cls, q):
        """TCP position for joints [j1..j5]; tool-down constraint j4 = -(j2+j3)."""
        j1, j2, j3 = q[0], q[1], q[2]
        j4 = -(j2 + j3)
        d = cls._ry(j4, [0.06, 0.0, -0.06]) + np.array([0.147, 0.0, 0.0])
        d = cls._ry(j3, d) + np.array([0.0, 0.0, 0.135])
        d = cls._ry(j2, d)
        c, s = np.cos(j1), np.sin(j1)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])

    def _q_from_env(self):
        return [s[0] for s in p.getJointStates(self.arm, self.movable)]

    def _tip_pose(self):
        ls = p.getLinkState(self.arm, self.link_idx, computeForwardKinematics=True)
        o = np.array(ls[4])
        R = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
        t = o + R @ CUP_LOCAL
        return np.array([t[0], t[1], t[2], 0.0])

    def _solve_ik(self, x, y, z):
        ls = p.getLinkState(self.arm, self.link_idx, computeForwardKinematics=True)
        o = np.array(ls[4])
        R = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
        tgt = (np.array([x, y, z]) - R @ CUP_LOCAL).tolist()
        q = p.calculateInverseKinematics(self.arm, self.link_idx, tgt, self._ik_target_orn,
                                         lowerLimits=self.lo, upperLimits=self.hi,
                                         restPoses=self._warm)
        return q

    def _refine_dls(self, target_xyz, iters=60):
        """Damped least squares on analytic FK; vars = (j1, j2, j3), j4 derived.

        j4 is wrapped to [-pi, pi] (continuous wrist joint: same tool-down pose).
        Multi-seed start points escape local minima.
        """
        tgt = np.asarray(target_xyz, dtype=float)
        lam2, h = 5e-4, 1e-4

        def f(u):
            q = [u[0], u[1], u[2]]
            return self._fk(q + [-(u[1] + u[2]), 0.0])

        lo_u = np.array([self.lo[i] for i in range(3)])
        hi_u = np.array([self.hi[i] for i in range(3)])
        cur = self._q_from_env()

        seeds = [np.array([cur[0], cur[1], cur[2]])]
        # a few generic elbow-up/down starts if the warm one stalls
        seeds += [np.array([s[0], s[1], s[2]]) for s in (
            (0.0, 0.6, -0.4), (0.0, 1.2, -1.0), (0.0, 0.3, 0.3),
            (np.arctan2(tgt[1], tgt[0]), 0.8, -0.6))]

        best_u, best_err = None, np.inf
        for u in seeds:
            u = np.clip(u, lo_u, hi_u)
            for _ in range(iters):
                e = tgt - f(u)
                if np.linalg.norm(e) < 0.0012:
                    break
                Jm = np.zeros((3, 3))
                for k in range(3):
                    up = u.copy(); up[k] += h
                    dn = u.copy(); dn[k] -= h
                    Jm[:, k] = (f(up) - f(dn)) / (2 * h)
                dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + lam2 * np.eye(3), e)
                step = float(np.max(np.abs(dq)))
                if step > 0.1:
                    dq *= 0.1 / step
                u = np.clip(u + dq, lo_u, hi_u)
            err = float(np.linalg.norm(f(u) - tgt))
            if err < best_err:
                best_u, best_err = u.copy(), err
            if err < 0.0012:
                break

        j4 = -(best_u[1] + best_u[2])
        j4 = (j4 + np.pi) % (2 * np.pi) - np.pi          # wrap to [-pi, pi]
        q_full = [float(best_u[0]), float(best_u[1]), float(best_u[2]), float(j4), 0.0]
        for j, qv in zip(self.movable, q_full):
            p.resetJointState(self.arm, j, qv)
        self._warm = list(q_full)
        return best_err < 0.0025, best_err

    def _move_to(self, x, y, z, iters=1):
        """Exact pybullet IK + enforced tool-down (j4). Iterates to absorb R5 coupling."""
        T = np.array([x, y, z])
        try:
            p.calculateInverseKinematics(self.arm, self.link_idx, [0, 0, 0],
                                         maxNumIterations=500)
            _ikw = {"maxNumIterations": 500}
        except TypeError:
            _ikw = {}
        for _ in range(2):
            ls = p.getLinkState(self.arm, self.link_idx, computeForwardKinematics=True)
            R = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
            tgt_o = (T - R @ CUP_LOCAL).tolist()
            q = p.calculateInverseKinematics(self.arm, self.link_idx, tgt_o, self._ik_target_orn,
                                             lowerLimits=self.lo, upperLimits=self.hi,
                                             restPoses=self._warm, **_ikw)
            qq = list(q)
            for j, qv in zip(self.movable, qq):
                p.resetJointState(self.arm, j, qv)
            self._warm = list(qq)
            if np.linalg.norm(self._tip_pose()[:3] - T) < 0.0015:
                break
        for _ in range(0):
            ls = p.getLinkState(self.arm, self.link_idx, computeForwardKinematics=True)
            R = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
            tgt_o = (T - R @ CUP_LOCAL).tolist()
            q = p.calculateInverseKinematics(self.arm, self.link_idx, tgt_o,
                                             lowerLimits=self.lo, upperLimits=self.hi,
                                             restPoses=self._warm, **_ikw)
            qq = list(q)
            for j, qv in zip(self.movable, qq):
                p.resetJointState(self.arm, j, qv)
            self._warm = list(qq)
            if np.linalg.norm(self._tip_pose()[:3] - T) < 0.0015:
                break
        # NOTE: no sim steps here (they let the undriven arm sag/jump between
        # positioning and pin; env.step() already steps physics after _move_to).
        if getattr(self, "carrying", False):
            self._attach_box()   # pin after stepping so carry never sags in renders
        tip = self._tip_pose()[:3]
        return bool(np.linalg.norm(tip - np.array([x, y, z])) < 0.015)

    # ---------- env API ----------
    def reset(self, color=None):
        self._load_scene()
        self.color = color or str(self.rng.choice(["red", "blue"]))
        self.side = {"red": "left", "blue": "right"}[self.color]
        bx = self.rng.uniform(*PICK_ZONE["x"])
        by = self.rng.uniform(*PICK_ZONE["y"])
        size = BOX_SIZE * self.rng.uniform(0.9, 1.1)
        col = list(COLORS[self.color])
        jitter = self.rng.uniform(-0.06, 0.06, 3)
        for i in range(3):
            col[i] = max(0.0, min(1.0, col[i] + jitter[i]))
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size / 2] * 3)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[size / 2] * 3, rgbaColor=col)
        self.box_size = size
        self.box = p.createMultiBody(size, shape, visual, [bx, by, TABLE_Z + size / 2])
        p.changeDynamics(self.box, -1, lateralFriction=float(self.rng.uniform(0.4, 0.9)))
        # no cup-vs-box contact dynamics: attachment is kinematic (pin), and the
        # collision mesh sits deeper than the visual cup; leaving it on would fight the pin
        p.setCollisionFilterPair(self.arm, self.box, 5, -1, 0)
        self.suction = 0
        self.carrying = False
        self._spawn_xy = np.array([bx, by])
        self._lifted = False
        self._assist_done = False
        self._armed = False
        self._since_attach = 10 ** 9
        self._was_carrying = False
        self._move_to(*HOME_XYZR[:3], iters=80)
        self.task = f"{self.color} box to {self.side} stack"
        return self.task

    def step(self, action):
        """action: [dx,dy,dz,dr,suction_target]; deltas clipped to 2cm/step."""
        a = np.asarray(action, dtype=float)
        cur = self._tip_pose()
        target = cur[:3] + np.clip(a[:3], -0.02, 0.02)
        rho = np.hypot(target[0], target[1])
        if rho > 0.26:
            target[:2] *= 0.26 / rho
        # suction assist: once suction is commanded while over the resting box,
        # press/align the cup down onto its top face (smooth <=3cm/step) so
        # attachment always happens ON contact; arming sticks until attach.
        self._assist = False
        if (getattr(self, "box", None) is not None and not self.carrying
                and not getattr(self, "_assist_done", False)):
            if int(round(float(np.clip(a[4], 0, 1)))) == 1 and not getattr(self, "_armed", False):
                bpos0 = np.array(p.getBasePositionAndOrientation(self.box)[0])
                if (abs(cur[0] - bpos0[0]) < self.box_size * 0.6
                        and abs(cur[1] - bpos0[1]) < self.box_size * 0.6):
                    self._armed = True
            if getattr(self, "_armed", False):
                bpos = np.array(p.getBasePositionAndOrientation(self.box)[0])
                btop = bpos[2] + self.box_size / 2
                if (abs(cur[0] - bpos[0]) < self.box_size * 0.6
                        and abs(cur[1] - bpos[1]) < self.box_size * 0.6):
                    contact_z = max(btop - 0.002, TABLE_Z + 0.01)
                    dz = cur[2] - contact_z
                    tz = contact_z if dz <= 0.03 else cur[2] - 0.03
                    tx = cur[0] + float(np.clip(bpos[0] - cur[0], -0.01, 0.01))
                    ty = cur[1] + float(np.clip(bpos[1] - cur[1], -0.01, 0.01))
                    target = np.array([tx, ty, tz])
                    self._assist = True
        target[2] = float(np.clip(target[2], TABLE_Z + 0.01, 0.30))
        ok = self._move_to(*target)
        for _ in range(5):
            p.stepSimulation()
        if (self.box is not None and np.array(
                p.getBasePositionAndOrientation(self.box)[0])[2]
                > TABLE_Z + self.box_size * 1.5):
            self._lifted = True
        self._suction(int(round(float(np.clip(a[4], 0, 1)))))
        was = getattr(self, "_was_carrying", False)
        if self.carrying:
            self._attach_box()   # re-pin AFTER stepping: the box is dynamic and would
            self._assist_done = True   # visibly sag a few mm before the render
            self._since_attach = 0 if not was else getattr(self, "_since_attach", 0) + 1
        else:
            self._since_attach = 10 ** 9
        self._was_carrying = self.carrying
        return self.get_obs(), not ok, {}

    def _attach_box(self):
        if getattr(self, "box", None) is None:
            return
        tip = self._tip_pose()[:3]
        p.resetBasePositionAndOrientation(
            self.box, [tip[0], tip[1], tip[2] + CUP_DROP - self.box_size / 2],
            p.getBasePositionAndOrientation(self.box)[1])

    def _in_contact(self):
        """Cup bottom is touching the box top face: vertical gap within (-15mm, +8mm)
        and xy inside the box footprint."""
        if getattr(self, "box", None) is None:
            return False
        tip = self._tip_pose()[:3]
        bpos = np.array(p.getBasePositionAndOrientation(self.box)[0])
        btop = bpos[2] + self.box_size / 2
        gap = (tip[2] + CUP_DROP) - btop
        return bool(-0.015 < gap < 0.008
                    and abs(tip[0] - bpos[0]) < self.box_size * 0.3
                    and abs(tip[1] - bpos[1]) < self.box_size * 0.3)

    def _suction(self, on):
        if on:
            # level-based latch: suction command stays armed, but the box only
            # attaches once the cup actually contacts its top face
            self.suction = 1
            if not self.carrying and self._in_contact():
                self.carrying = True
                self._attach_box()
        else:
            self.carrying = False
            self.suction = 0

    def get_obs(self):
        tip = self._tip_pose()
        return np.array([tip[0], tip[1], tip[2], tip[3], float(self.suction)])

    def render(self, jitter=False, dist=None, target=None):
        pitch, yaw = self.cam_pitch, self.cam_yaw
        if dist is None:
            dist = self.cam_dist
        if jitter:
            dist += self.rng.uniform(-0.05, 0.05)
            pitch += self.rng.uniform(-4, 4)
            yaw += self.rng.uniform(-6, 6)
        target = [0.16, 0, 0] if target is None else list(target)
        view = p.computeViewMatrixFromYawPitchRoll(target, dist, yaw, pitch, 0, 2)
        proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
        w = h = self.render_size
        try:
            _, _, rgb, _, _ = p.getCameraImage(w * 2, h * 2, view, proj,
                                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
            img = rgb[:, :, :3].astype(np.uint8)
            import cv2
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        except Exception:
            _, _, rgb, _, _ = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER)
            img = rgb[:, :, :3].astype(np.uint8)
        return img

    def success(self):
        if self.box is None:
            return False
        bpos = np.array(p.getBasePositionAndOrientation(self.box)[0])
        s = STACKS[self.side]
        near_xy = abs(bpos[0] - s["x"]) < 0.04 and abs(bpos[1] - s["y"]) < 0.04
        return bool(getattr(self, "_lifted", False) and near_xy)

    def close(self):
        p.disconnect()


# ---------------- scripted expert (privileged state) ----------------
def run_expert(env, max_steps=160):
    """Scripted pick-place using privileged box pose; returns trajectory frames."""
    bx, by, _ = p.getBasePositionAndOrientation(env.box)[0]
    s = STACKS[env.side]
    z_top = TABLE_Z + env.box_size          # top face of the box
    traj = []

    def move(x, y, z):
        for _ in range(40):
            err = np.array([x, y, z]) - env._tip_pose()[:3]
            if np.linalg.norm(err) < 0.006:
                break
            step = np.clip(err, -0.025, 0.025)
            a = np.array([step[0], step[1], step[2], 0.0, float(env.suction)])
            obs, _, _ = env.step(a)
            traj.append((obs.copy(), a.copy(), env.render()))

    move(bx, by, z_top + 0.05)              # above box
    move(bx, by, z_top - 0.002)             # press cup onto top face (true contact)
    for _ in range(3):                      # dwell low so policy learns deep approach
        move(bx, by, z_top - 0.002)
    env._suction(1)
    traj[-1] = (env.get_obs().copy(), np.array([0, 0, 0, 0, 1.0]), env.render())
    move(bx, by, TABLE_Z + 0.12)            # lift
    move(s["x"], s["y"], TABLE_Z + 0.13)    # over stack zone
    move(s["x"], s["y"], TABLE_Z + env.box_size)  # place: cup at top-face height => box rests on table
    env._suction(0)
    traj[-1] = (env.get_obs().copy(), np.array([0, 0, 0, 0, 0.0]), env.render())
    move(s["x"], s["y"], TABLE_Z + 0.13)    # retreat up
    return traj

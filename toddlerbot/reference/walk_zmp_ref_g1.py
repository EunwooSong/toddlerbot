"""Unitree G1 variants of the ZMP walk reference.

G1 differs from Toddlerbot in: joint sign conventions (no left/right motor
mirroring), no neck, direct-drive 3-DoF waist (frozen), static arms
(forward-90 pose baked into default_pos), serial ankle, gear ratio 1.

Leg-vector convention (matches descriptions/unitree_g1 config order):
  [hip_pitch, hip_roll, hip_yaw, knee, ank_roll, ank_pitch] x (left, right)

Sign tables are FROZEN from FK brute-force validation
(unitree_g1/validate_g1_ik.py) - do not hand-edit.
"""
import os
import pickle

import mujoco
import numpy

from toddlerbot.algorithms.zmp_walk import ZMPWalk
from toddlerbot.reference.walk_zmp_ref import WalkZMPReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np


class ZMPWalkG1(ZMPWalk):
    """ZMPWalk with G1 joint-sign conventions in foot_ik."""

    # [hip_pitch, hip_roll, hip_yaw, knee, ank_roll(hr term, ar term), ank_pitch]
    # FROZEN by unitree_g1/validate_g1_ik.py FK brute force (2026-08-06):
    # identical for both sides (G1 joints are not left/right mirrored).
    # FK residuals: <13 mm without yaw command, <38 mm at max turn (tilted,
    # offset hip-yaw axis modeled to first order only).
    SIGN = {
        "left": {"hp": -1.0, "hr": -1.0, "hy": 1.0, "knee": 1.0,
                 "ar_hr": 1.0, "ar_o": 1.0, "ap": -1.0},
        "right": {"hp": -1.0, "hr": -1.0, "hy": 1.0, "knee": 1.0,
                  "ar_hr": 1.0, "ar_o": 1.0, "ap": -1.0},
    }

    def foot_ik(self, target_foot_pos, target_foot_ori, side="left"):
        target_x = target_foot_pos[:, 0]
        target_y = target_foot_pos[:, 1]
        target_z = target_foot_pos[:, 2]
        ank_roll = target_foot_ori[:, 0]
        ank_pitch = target_foot_ori[:, 1]
        hip_yaw = target_foot_ori[:, 2]

        offsets = self.robot.config["general"]["offsets"]

        # G1's hip_yaw axis is laterally/forward offset from hip_pitch, so the
        # yaw rotation must be taken about that axis position, not the origin.
        # (The 10-deg yaw-axis tilt is NOT modeled: ~10 mm / 3 deg residual at
        # max turn command — acceptable for a soft reference.)
        ax = offsets.get("hip_yaw_offset_x", 0.0)
        ay = offsets.get("hip_yaw_offset_y", 0.0) * (1.0 if side == "left" else -1.0)
        px, py = target_x - ax, target_y - ay
        transformed_x = px * np.cos(hip_yaw) + py * np.sin(hip_yaw) + ax
        transformed_y = -(-px * np.sin(hip_yaw) + py * np.cos(hip_yaw) + ay)
        transformed_z = (
            offsets["hip_pitch_to_knee_z"]
            + offsets["knee_to_ank_pitch_z"]
            - target_z
            - self.default_target_z
        )

        hip_roll = np.arctan2(
            transformed_y, transformed_z + offsets["hip_roll_to_pitch_z"]
        )

        leg_projected_yz_length = np.sqrt(transformed_y**2 + transformed_z**2)
        leg_length = np.sqrt(transformed_x**2 + leg_projected_yz_length**2)
        leg_pitch = np.arctan2(transformed_x, leg_projected_yz_length)
        hip_disp_cos = (
            leg_length**2
            + offsets["hip_pitch_to_knee_z"] ** 2
            - offsets["knee_to_ank_pitch_z"] ** 2
        ) / (2 * leg_length * offsets["hip_pitch_to_knee_z"])
        hip_disp = np.arccos(np.clip(hip_disp_cos, -1.0, 1.0))
        ank_disp = np.arcsin(
            np.clip(
                offsets["hip_pitch_to_knee_z"]
                / offsets["knee_to_ank_pitch_z"]
                * np.sin(hip_disp),
                -1.0,
                1.0,
            )
        )
        hip_pitch = leg_pitch + hip_disp
        knee_pitch = hip_disp + ank_disp
        ank_pitch = ank_pitch + knee_pitch - hip_pitch

        # G1's hip_yaw axis is tilted ~10 deg in the xz-plane: scale the joint
        # angle to get the commanded z-yaw, and cancel the parasitic roll with
        # hip_roll (first-order; residual ~10 mm at max turn).
        ux = offsets.get("hip_yaw_axis_x", 0.0)
        uz = offsets.get("hip_yaw_axis_z", 1.0)
        hy_joint = hip_yaw / uz
        hr_comp = (ux / uz) * hip_yaw

        s = self.SIGN[side]
        return np.vstack(
            [
                s["hp"] * hip_pitch,
                s["hr"] * hip_roll - hr_comp,
                s["hy"] * hy_joint,
                s["knee"] * knee_pitch,
                s["ar_hr"] * hip_roll + s["ar_o"] * ank_roll,
                s["ap"] * ank_pitch,
            ]
        ).T


class WalkZMPReferenceG1(WalkZMPReference):
    """WalkZMPReference for Unitree G1: static upper body, G1 kinematics."""

    def _setup_neck(self):
        pass  # G1 has no neck; overridden get_state_ref never touches it

    def _setup_arm(self):
        pass  # arms frozen at default_pos (forward-90); no dataset needed

    def _setup_waist(self):
        pass  # waist frozen at 0; no closed-loop coefficients

    def _setup_zmp(self):
        # leg-vector positions of hip_roll (for waist-roll compensation; waist
        # is frozen on G1 so these adds are no-ops, kept for parent parity)
        self.left_hip_roll_rel_idx = 1
        self.right_hip_roll_rel_idx = 7

        # G1's ankle ranges are small (roll +-0.26, pitch -0.87..0.52): the
        # flat-foot IK saturates them at command extremes (|vx|=1, |vy|=0.5).
        # Clip the leg reference to joint limits — the foot-orientation ref
        # gives up flatness at extreme strides (the kinematic analogue of
        # heel/toe lift) instead of pulling the policy past its limits.
        leg_names = [n for n in self.robot.joint_ordering
                     if self.robot.joint_groups[n] == "leg"]
        self.leg_limit_lo = np.array(
            [self.robot.joint_limits[n][0] for n in leg_names], dtype=np.float32)
        self.leg_limit_hi = np.array(
            [self.robot.joint_limits[n][1] for n in leg_names], dtype=np.float32)

        single_double_ratio = 2.0
        self.zmp_walk = ZMPWalkG1(self.robot, self.cycle_time, single_double_ratio)
        self.single_support_ratio = single_double_ratio / (single_double_ratio + 1)
        self.double_support_ratio = 1 - self.single_support_ratio

        lookup_table_path = os.path.join(
            "toddlerbot", "descriptions", self.robot.name, "walk_zmp_lookup_table.pkl"
        )
        if os.path.exists(lookup_table_path):
            with open(lookup_table_path, "rb") as f:
                (
                    lookup_keys,
                    com_ref_list,
                    stance_mask_ref_list,
                    leg_joint_pos_ref_list,
                ) = pickle.load(f)
        else:
            # G1 command grid — MUST cover walk_g1.gin command_range[5:8]
            # (parent default is Toddlerbot-scale vx<=0.4: out-of-grid commands
            # would snap to the nearest key and de-sync gait from the velocity
            # reward target). Coarser 0.05 interval keeps the table compact.
            lookup_keys, com_ref_list, leg_joint_pos_ref_list, stance_mask_ref_list = (
                self.zmp_walk.build_lookup_table(
                    command_range=[[-1.0, 1.0], [-0.5, 0.5], [-1.0, 1.0]],
                    interval=0.05,
                )
            )
            with open(lookup_table_path, "wb") as f:
                pickle.dump(
                    (
                        lookup_keys,
                        com_ref_list,
                        stance_mask_ref_list,
                        leg_joint_pos_ref_list,
                    ),
                    f,
                )

        self.lookup_keys = np.array(lookup_keys, dtype=np.float32)
        self.lookup_length = np.array(
            [len(m) for m in stance_mask_ref_list], dtype=np.float32
        )
        num_max = max(len(m) for m in stance_mask_ref_list)
        stance = numpy.zeros((len(stance_mask_ref_list), num_max, 2), dtype=numpy.float32)
        legpos = numpy.zeros((len(stance_mask_ref_list), num_max, 12), dtype=numpy.float32)
        for i, (mask, pos) in enumerate(zip(stance_mask_ref_list, leg_joint_pos_ref_list)):
            stance[i, : len(mask)] = mask
            legpos[i, : len(pos)] = pos
        self.stance_mask_lookup = np.asarray(stance)
        self.leg_joint_pos_lookup = np.asarray(legpos)

        if self.use_jax:
            import jax

            self.lookup_keys = jax.device_put(self.lookup_keys)
            self.lookup_length = jax.device_put(self.lookup_length)
            self.stance_mask_lookup = jax.device_put(self.stance_mask_lookup)
            self.leg_joint_pos_lookup = jax.device_put(self.leg_joint_pos_lookup)

    def _setup_mjx(self, com_z_lower_limit_offset: float = 0.01):
        from mujoco import mjx

        from toddlerbot.utils.file_utils import find_robot_file_path

        xml_path = find_robot_file_path(self.robot.name, suffix="_scene.xml")
        model = mujoco.MjModel.from_xml_path(xml_path)
        self.default_qpos = np.array(model.keyframe("home").qpos)
        self.mj_joint_indices = (
            np.array(
                [
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    for name in self.robot.joint_ordering
                ]
            )
            - 1
        )
        self.mj_motor_indices = self.mj_joint_indices  # 1:1, no transmissions
        self.mj_passive_indices = np.array([], dtype=int)
        self.passive_joint_indices = np.array([], dtype=int)
        self.passive_joint_signs = np.array([], dtype=np.float32)

        self.left_foot_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_center"
        )
        self.right_foot_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_center"
        )

        def bid(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

        hip_pitch_id = bid("left_hip_pitch_link")
        hip_roll_id = bid("left_hip_roll_link")
        knee_id = bid("left_knee_link")
        ank_pitch_id = bid("left_ankle_pitch_link")
        ank_roll_id = bid("left_ank_roll_link")

        if self.use_jax:
            self.model = mjx.put_model(model)

            def forward(qpos):
                data = mjx.make_data(self.model)
                data = data.replace(qpos=qpos)
                return mjx.forward(self.model, data)

        else:
            self.model = model

            def forward(qpos):
                data = mujoco.MjData(self.model)
                data.qpos = qpos
                mujoco.mj_forward(self.model, data)
                return data

        self.forward = forward

        data = self.forward(self.default_qpos)
        self.left_foot_center = np.asarray(data.site_xpos[self.left_foot_site_id])
        self.right_foot_center = np.asarray(data.site_xpos[self.right_foot_site_id])
        self.torso_pos_init = np.asarray(data.qpos[:3])
        self.desired_com = (self.left_foot_center + self.right_foot_center) / 2.0

        self.knee_default = self.default_joint_pos[self.left_knee_idx]
        self.knee_max = np.max(
            np.abs(np.array(self.robot.joint_limits["left_knee"], dtype=np.float32))
        )

        d_hip_knee = np.asarray(data.xpos[hip_pitch_id] - data.xpos[knee_id])
        self.hip_to_knee_len = np.sqrt(d_hip_knee[0] ** 2 + d_hip_knee[2] ** 2)
        d_knee_ank = np.asarray(data.xpos[knee_id] - data.xpos[ank_pitch_id])
        self.knee_to_ank_len = np.sqrt(d_knee_ank[0] ** 2 + d_knee_ank[2] ** 2)

        hip_to_ank_pitch_default = np.asarray(
            data.xpos[hip_pitch_id] - data.xpos[ank_pitch_id], dtype=np.float32
        )
        hip_to_ank_roll_default = np.asarray(
            data.xpos[hip_roll_id] - data.xpos[ank_roll_id], dtype=np.float32
        )
        self.hip_to_ank_pitch_default = inplace_update(hip_to_ank_pitch_default, 1, 0.0)
        self.hip_to_ank_roll_default = inplace_update(hip_to_ank_roll_default, 0, 0.0)

        self.com_z_limits = np.array(
            [self.com_fk(self.knee_max)[2] + com_z_lower_limit_offset, 0.0],
            dtype=np.float32,
        )

    def com_ik(self, com_z, com_x=None, com_y=None):
        """G1-signed com_ik: recover raw angles from the parent's Toddlerbot
        vector [hp, hr, 0, -knee, hr, -ap] x mirrored, re-emit in G1
        convention (frozen sign table; both legs identical, roll physically
        shared: hip=-hr_raw, ankle=+hr_raw keeps the foot flat)."""
        v = super().com_ik(com_z, com_x, com_y)
        leg6 = np.array([-v[0], -v[1], 0.0, -v[3], v[1], v[5]], dtype=np.float32)
        return np.concatenate([leg6, leg6])

    def get_state_ref(
        self, state_curr: ArrayType, time_curr, command: ArrayType
    ) -> ArrayType:
        path_state = self.integrate_path_state(state_curr, command)

        # upper body: frozen at default (arms forward-90, waist 0) - already
        # baked into default_joint_pos/default_motor_pos from the robot config
        joint_pos = self.default_joint_pos.copy()
        motor_pos = self.default_motor_pos.copy()

        is_static_pose = np.logical_or(
            np.linalg.norm(command[5:]) < 1e-6, time_curr < 1e-6
        )
        nearest_command_idx = np.argmin(
            np.linalg.norm(self.lookup_keys - command[5:], axis=1)
        )
        step_idx = np.round(time_curr / self.dt).astype(int)

        def get_leg_joint_pos_init() -> ArrayType:
            state_ref = np.concatenate((path_state, motor_pos, joint_pos))
            qpos = self.get_qpos_ref(state_ref)
            data = self.forward(qpos)
            com_pos = np.array(data.subtree_com[0], dtype=np.float32)
            com_pos_error = self.desired_com[:2] - com_pos[:2]
            com_ctrl = self.com_kp * com_pos_error
            return self.com_ik(0, com_ctrl[0], com_ctrl[1])

        def get_leg_joint_pos() -> ArrayType:
            return self.leg_joint_pos_lookup[nearest_command_idx][
                (step_idx % self.lookup_length[nearest_command_idx]).astype(int)
            ]

        from toddlerbot.utils.array_utils import conditional_update

        leg_joint_pos = conditional_update(
            is_static_pose, get_leg_joint_pos_init, get_leg_joint_pos
        )
        leg_joint_pos = np.clip(leg_joint_pos, self.leg_limit_lo, self.leg_limit_hi)
        joint_pos = inplace_update(joint_pos, self.leg_joint_indices, leg_joint_pos)
        motor_pos = inplace_update(motor_pos, self.leg_motor_indices, leg_joint_pos)

        stance_mask = np.where(
            is_static_pose,
            np.ones(2, dtype=np.float32),
            self.stance_mask_lookup[nearest_command_idx][
                (step_idx % self.lookup_length[nearest_command_idx]).astype(int)
            ],
        )

        return np.concatenate((path_state, motor_pos, joint_pos, stance_mask))

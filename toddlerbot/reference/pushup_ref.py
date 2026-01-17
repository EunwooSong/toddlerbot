import os
import joblib
import numpy
from typing import Tuple, List, Optional

from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import array_lib as np
from toddlerbot.utils.array_utils import ArrayType

from toddlerbot.reference.motion_ref import MotionReference

class PushupReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        dt: float,
        run_name: str = "push_up",
        com_kp: List[float] = [1.0, 1.0],
    ):
        super().__init__("push_up", run_name, robot, dt, com_kp)
        
        # 1. 데이터 로딩 (ReplayPolicy 로직 적용)
        motion_file_path = os.path.join("motion", f"{run_name}.pkl")
        
        if os.path.exists(motion_file_path):
            data_dict = joblib.load(motion_file_path)
            
            # 파일 포맷에 따른 키 값 처리 ('action_traj' or 'qpos')
            raw_action_arr = np.array(
                data_dict.get("action_traj", data_dict.get("qpos")), 
                dtype=np.float32
            )
            
            # 2. 로봇 하드웨어(Gripper 유무)와 데이터 Shape 불일치 처리
            robot_dim = len(robot.motor_ordering)
            data_dim = raw_action_arr.shape[1]

            if robot.has_gripper and data_dim < robot_dim:
                padding = np.zeros((raw_action_arr.shape[0], robot_dim - data_dim), dtype=np.float32)
                raw_action_arr = np.concatenate([raw_action_arr, padding], axis=1)
            elif not robot.has_gripper and data_dim > robot_dim:
                raw_action_arr = raw_action_arr[:, :robot_dim]
            
            self.motion_data = raw_action_arr
        else:
            print(f"Warning: {motion_file_path} not found. Using default pose.")
            self.motion_data = np.tile(self.default_motor_pos, (100, 1))

        self.n_frames = self.motion_data.shape[0]

    def get_phase_signal(self, time_curr: float | ArrayType, init_idx: int = 0) -> ArrayType:
        """Get the phase signal for the current time."""
        # Calculate the index based on time and init_idx
        time_idx = np.floor(time_curr / self.dt).astype(np.int32)
        total_idx = (init_idx + time_idx) % self.n_frames

        # Calculate phase based on total_idx
        phase = (total_idx / self.n_frames) * 2 * np.pi
        phase_signal = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)

        return phase_signal

    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        """
        제자리 운동이므로 목표 속도는 0으로 설정합니다.
        """
        lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        ang_vel = np.zeros(3, dtype=np.float32)
        return lin_vel, ang_vel

    def get_state_ref(
        self, state_curr: ArrayType, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        """
        현재 시간에 해당하는 Reference State를 반환합니다.
        JAX 호환성을 위해 .at[].set()을 사용하여 배열을 업데이트합니다.
        """
        # 1. 현재 프레임 인덱스 계산 (Cyclic)
        time_idx = np.floor(time_curr / self.dt).astype(np.int32)
        current_frame_idx = time_idx % self.n_frames
        
        # 2. Reference Motor Position 추출
        ref_motor_pos = self.motion_data[current_frame_idx]

        # 3. Motor Position -> Joint Position 변환 (FK & Gear Ratio)
        # JAX 배열 초기화
        ref_joint_pos = np.zeros(self.robot.nu, dtype=np.float32)
        
        # [수정됨] In-place assignment 대신 .at[].set() 사용
        # 각 set() 호출은 새로운 배열을 반환하므로 ref_joint_pos를 계속 갱신해야 합니다.
        
        # Neck
        ref_joint_pos = ref_joint_pos.at[self.neck_joint_indices].set(
            self.neck_fk(ref_motor_pos[self.neck_motor_indices])
        )
        
        # Arm
        ref_joint_pos = ref_joint_pos.at[self.arm_joint_indices].set(
            self.arm_fk(ref_motor_pos[self.arm_motor_indices])
        )
        
        # Waist
        ref_joint_pos = ref_joint_pos.at[self.waist_joint_indices].set(
            self.waist_fk(ref_motor_pos[self.waist_motor_indices])
        )
        
        # Leg
        ref_joint_pos = ref_joint_pos.at[self.leg_joint_indices].set(
            self.leg_fk(ref_motor_pos[self.leg_motor_indices])
        )

        # 4. Path State (Root Body) 설정
        path_pos = state_curr[:3]
        path_quat = state_curr[3:7]
        path_lin_vel = np.zeros(3, dtype=np.float32) 
        path_ang_vel = np.zeros(3, dtype=np.float32)
        
        path_state_ref = np.concatenate([path_pos, path_quat, path_lin_vel, path_ang_vel])

        # 5. 최종 반환 벡터 결합
        full_state_ref = np.concatenate(
            [path_state_ref, ref_motor_pos, ref_joint_pos], axis=0
        )
        
        return full_state_ref
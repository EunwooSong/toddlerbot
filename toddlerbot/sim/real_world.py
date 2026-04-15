import platform
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np

from toddlerbot.actuation import JointState
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_ports

# feat: heat2torque 결합, old
# from heat2torque.envs.tjx_env import ThermalJaxEnv
# from heat2torque.utils.actuator import matching_actuator_config, load_motor_params_from_csv
# from heat2torque.envs.config import MTJXConfig
# from heat2torque.envs.base import HeatState
# 데이터를 받아 분석하기 위해, 처리를 해야 함.

from heat2torque.config import ThermalConfig
from heat2torque.obs import RealThermalObs

import time
# dt 계산용

class RealWorld(BaseSim):
    """Real-world robot interface class."""

    def __init__(self, robot: Robot):
        """Initializes the real-world robot interface.

        Args:
            robot (Robot): An instance of the Robot class containing configuration details.

        Attributes:
            has_imu (bool): Indicates if the robot is equipped with an Inertial Measurement Unit (IMU).
            has_dynamixel (bool): Indicates if the robot uses Dynamixel motors.
            negated_motor_names (List[str]): A list of motor names that require direction negation due to URDF configuration issues.
        """
        super().__init__("real_world")
        self.robot = robot

        self.has_imu = self.robot.config["general"]["has_imu"]
        self.has_dynamixel = self.robot.config["general"]["has_dynamixel"]

        # TODO: Fix the mate directions in the URDF and remove the negated_motor_names
        self.negated_motor_names: List[str] = [
            "neck_pitch_act",
            "left_sho_roll",
            "right_sho_roll",
            "left_elbow_roll",
            "right_elbow_roll",
            "left_wrist_pitch_drive",
            "right_wrist_pitch_drive",
            "left_gripper_rack",
            "right_gripper_rack",
        ]

        # # init tjx env
        # acts = matching_actuator_config(self.robot)
        # # parameter 불러와서 적용
        # #load_motor_params_from_csv(acts)
        # # cfg: no, DR
        # cfg = MTJXConfig()
        # # 아래 내용도 나중에 arguments로 받아야함
        # cfg.tjx_cfg.use_basic_obs = True
        # cfg.tjx_cfg.use_derate = True
        # self.tjx_env = ThermalJaxEnv(acts, cfg=cfg)
        # self.tjx_cfg = cfg.tjx_cfg

        # if self.tjx_cfg.olaf:
        #     cfg.tjx_cfg.obs_clip = [60.0, 80.0]

        # # 초기 heat_state 설정
        # # 주변 온도는 cfg로 주어졌다고 가정
        # # 모터 초기 온도는 향후 주어진다고 가정
        # # 모터 초기 온도의 housing, core 온도는 동일하다고 가정
        # self.heat_state = HeatState(
        #     np.zeros(len(acts)),
        #     np.full((len(acts), ), cfg.tjx_cfg.c_a),
        #     np.full((len(acts), ), cfg.tjx_cfg.c_a),
        #     np.zeros(len(acts)),
        #     self.tjx_env.overheat,
        # )

        # real에서는 오직 관측값에 대해서만 처리
        

        # 만약, sim에서 시뮬레이션도 필요하다면, 그때가서 처리

        # 이전 통신 시간
        self.prev_comm_time = None
        self.initialize()

    def initialize(self) -> None:
        """Initializes the robot's components, including IMU and Dynamixel controllers, if available.

        This method sets up a thread pool executor to initialize the IMU and Dynamixel controllers asynchronously. It checks the operating system type to determine the appropriate port description for Dynamixel communication. If the robot is configured with an IMU, it initializes the IMU in a separate thread. Similarly, if the robot has Dynamixel actuators, it configures and initializes the Dynamixel controller using the specified port, baud rate, and control parameters. After initialization, it retrieves the results of the asynchronous operations and assigns them to the respective attributes. Finally, it performs a series of observations to ensure the components are functioning correctly.
        """
        self.executor = ThreadPoolExecutor()

        os_type = platform.system()

        future_imu = None
        if self.has_imu:
            from toddlerbot.sensing.IMU import IMU

            future_imu = self.executor.submit(IMU)

        future_dynamixel = None
        if self.has_dynamixel:
            from toddlerbot.actuation.dynamixel_control import (
                DynamixelConfig,
                DynamixelController,
            )

            description = (
                "USB Serial Port"
                if os_type == "Windows"
                else "USB <-> Serial Converter"
            )

            dynamixel_ports: List[str] = find_ports(description)

            dynamixel_ids = self.robot.get_joint_attrs("type", "dynamixel", "id")
            dynamixel_config = DynamixelConfig(
                port=dynamixel_ports[0],
                baudrate=self.robot.config["general"]["dynamixel_baudrate"],
                control_mode=self.robot.get_joint_attrs(
                    "type", "dynamixel", "control_mode"
                ),
                kP=self.robot.get_joint_attrs("type", "dynamixel", "kp_real"),
                kI=self.robot.get_joint_attrs("type", "dynamixel", "ki_real"),
                kD=self.robot.get_joint_attrs("type", "dynamixel", "kd_real"),
                kFF2=self.robot.get_joint_attrs("type", "dynamixel", "kff2_real"),
                kFF1=self.robot.get_joint_attrs("type", "dynamixel", "kff1_real"),
                init_pos=self.robot.get_joint_attrs("type", "dynamixel", "init_pos"),
            )
            future_dynamixel = self.executor.submit(
                DynamixelController, dynamixel_config, dynamixel_ids
            )

        if future_dynamixel is not None:
            self.dynamixel_controller = future_dynamixel.result()
        if future_imu is not None:
            try:
                self.imu = future_imu.result()
            except Exception as e:
                print(e)
                self.has_imu = False

        obs = None
        for _ in range(100):
            obs = self.get_observation()

        #self.thermal_env.reset()

        # # TODO 측정된 초기 모터 온도값으로, heat_state 초기화
        # self.heat_state = self.heat_state.replace(
        #     st_t_core = obs.motor_temp,
        #     st_t_housing = obs.motor_temp,
        # )
        # # 루프 시작 전 기준 시점을 현재 obs의 시간으로 설정
        # self.prev_comm_time = obs.time

    # @profile()
    def process_motor_reading(self, results: Dict[str, Dict[int, JointState]]) -> Obs:
        """Processes motor readings and returns an observation object.

        Args:
            results (Dict[str, Dict[int, JointState]]): A dictionary containing motor state data, indexed by motor type and ID.

        Returns:
            Obs: An observation object containing the current time, motor positions, velocities, and torques.
        """
        motor_state_dict_unordered: Dict[str, JointState] = {}
        if self.has_dynamixel:
            dynamixel_state = results["dynamixel"]
            for motor_name in self.robot.get_joint_attrs("type", "dynamixel"):
                motor_id = self.robot.config["joints"][motor_name]["id"]
                motor_state_dict_unordered[motor_name] = dynamixel_state[motor_id]

        time_curr = 0.0
        motor_pos = np.zeros(len(self.robot.motor_ordering), dtype=np.float32)
        motor_vel = np.zeros(len(self.robot.motor_ordering), dtype=np.float32)
        motor_tor = np.zeros(len(self.robot.motor_ordering), dtype=np.float32)

        # feat: 모터 온도 정보 추가
        motor_temp = np.zeros(len(self.robot.motor_ordering), dtype=np.float32)

        for i, motor_name in enumerate(self.robot.motor_ordering):
            if i == 0:
                time_curr = motor_state_dict_unordered[motor_name].time

            if motor_name in self.negated_motor_names:
                motor_pos[i] = -motor_state_dict_unordered[motor_name].pos
                motor_vel[i] = -motor_state_dict_unordered[motor_name].vel
            else:
                motor_pos[i] = motor_state_dict_unordered[motor_name].pos
                motor_vel[i] = motor_state_dict_unordered[motor_name].vel

            motor_tor[i] = abs(motor_state_dict_unordered[motor_name].tor)

            # feat: 배열에 모터 온도 추가
            motor_temp[i] = motor_state_dict_unordered[motor_name].temp

        # TODO: 여기서 obs에 온도 값 추가하면 됨
        
        obs = Obs(
            time=time_curr,
            motor_pos=motor_pos,
            motor_vel=motor_vel,
            motor_tor=motor_tor,
            motor_temp=motor_temp,
        )
        return obs

    def step(self):
        # 여기서 온도 값?
        pass

    # @profile()
    def get_observation(self, retries: int = 0):
        """Retrieve and process sensor observations asynchronously.

        This method collects data from available sensors, such as Dynamixel motors and IMU, using asynchronous calls. It processes the collected data to generate a comprehensive observation object.

        Args:
            retries (int, optional): The number of retry attempts for obtaining motor state data. Defaults to 0.

        Returns:
            An observation object containing processed sensor data, including motor states and, if available, IMU angular velocity and Euler angles.
        """
        results: Dict[str, Any] = {}
        futures: Dict[str, Any] = {}
        if self.has_dynamixel:
            # results["dynamixel"] = self.dynamixel_controller.get_motor_state(retries)
            futures["dynamixel"] = self.executor.submit(
                self.dynamixel_controller.get_motor_state, retries
            )

        if self.has_imu:
            # results["imu"] = self.imu.get_state()
            futures["imu"] = self.executor.submit(self.imu.get_state)

        # start_times = {key: time.time() for key in futures.keys()}
        for future in as_completed(futures.values()):
            for key, f in futures.items():
                if f is future:
                    # end_time = time.time()
                    results[key] = future.result()
                    # log(f"Time taken for {key}: {end_time - start_times[key]}", header=snake2camel(self.name), level="debug")
                    break

        obs = self.process_motor_reading(results)

        # if self.prev_comm_time is not None:
        #     # 현재 통신 시간과 이전 통신 시간의 차이 계산
        #     measured_dt = obs.time - self.prev_comm_time
            
        #     # 통신 지터나 에러로 인한 비정상적인 dt 방지 (최대 0.1초 제한)
        #     # 기본 dt(1/50)를 fallback으로 사용하거나 clip 적용
        #     dt_to_use = np.clip(measured_dt, 1e-4, 0.1)
        # else:
        #     # 초기 실행 시에는 기본 설정된 dt 사용
        #     dt_to_use = 1/50.0

        # 여기서 온도 계산 진행
        # self.heat_state = self.tjx_env.step_real_thermal(
        #     self.heat_state,
        #     obs.motor_tor,
        #     dt_to_use
        # )

        # 다음 스텝 시간 저장
        self.prev_comm_time = obs.time

        if self.has_imu:
            obs.ang_vel = np.array(results["imu"]["ang_vel"], dtype=np.float32)
            obs.euler = np.array(results["imu"]["euler"], dtype=np.float32)

        return obs


    # @profile()
    def set_motor_target(self, motor_angles: Dict[str, float]):
        """Sets the target angles for the robot's motors, adjusting for any negated motor directions and updating the positions of Dynamixel motors if present.

        Args:
            motor_angles (Dict[str, float]): A dictionary mapping motor names to their target angles in degrees.
        """

        # Directions are tuned to match the assembly of the robot.
        motor_angles_updated: Dict[str, float] = {}
        for name, angle in motor_angles.items():
            if name in self.negated_motor_names:
                motor_angles_updated[name] = -angle
            else:
                motor_angles_updated[name] = angle

        if self.has_dynamixel:
            dynamixel_pos = [
                motor_angles_updated[k]
                for k in self.robot.get_joint_attrs("type", "dynamixel")
            ]
            self.executor.submit(self.dynamixel_controller.set_pos, dynamixel_pos)

    def set_motor_kps(self, motor_kps: Dict[str, float]):
        """Sets the proportional gain (Kp) values for motors of type 'dynamixel'.

        If the robot has Dynamixel motors, this method updates their Kp values based on the provided dictionary. If a motor's Kp is not specified in the dictionary, it defaults to the value in the robot's configuration.

        Args:
            motor_kps (Dict[str, float]): A dictionary mapping motor names to their desired Kp values.
        """

        if self.has_dynamixel:
            dynamixel_kps: List[float] = []
            for k in self.robot.get_joint_attrs("type", "dynamixel"):
                if k in motor_kps:
                    dynamixel_kps.append(motor_kps[k])
                else:
                    dynamixel_kps.append(self.robot.config["joints"][k]["kp_real"])

            self.executor.submit(self.dynamixel_controller.set_kp, dynamixel_kps)

    def close(self):
        """Closes all active components and shuts down the executor.

        This method checks for active components such as Dynamixel motors and IMU sensors. If they are present, it submits tasks to close them using the executor. Finally, it shuts down the executor, ensuring all submitted tasks are completed before termination.
        """

        if self.has_dynamixel:
            self.executor.submit(self.dynamixel_controller.close_motors)
        if self.has_imu:
            self.executor.submit(self.imu.close)

        self.executor.shutdown(wait=True)

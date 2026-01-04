import time
import matplotlib.pyplot as plt
import numpy as np
import csv  # 데이터 저장을 위한 모듈 추가
from toddlerbot.sensing.IMU import IMU

if __name__ == "__main__":
    imu = IMU()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("IMU Readings - Euler Angles and Angular Velocities")

    # [그래프 설정 생략 - 기존과 동일]
    ax1.set_title("Euler Angles (Radians)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angle (rad)")
    line_roll, = ax1.plot([], [], label="Roll", color="r")
    line_pitch, = ax1.plot([], [], label="Pitch", color="g")
    line_yaw, = ax1.plot([], [], label="Yaw", color="b")
    ax1.legend(); ax1.set_ylim(-np.pi, np.pi)

    ax2.set_title("Angular Velocity (Rad/s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angular Velocity (rad/s)")
    line_ang_x, = ax2.plot([], [], label="Ang Vel X", color="r")
    line_ang_y, = ax2.plot([], [], label="Ang Vel Y", color="g")
    line_ang_z, = ax2.plot([], [], label="Ang Vel Z", color="b")
    ax2.legend(); ax2.set_ylim(-5, 5)

    euler_data = {"time": [], "roll": [], "pitch": [], "yaw": []}
    ang_vel_data = {"time": [], "x": [], "y": [], "z": []}
    start_time = time.time()

    def update_plot():
        state = imu.get_state()
        current_time = time.time() - start_time

        # 데이터 저장
        euler_data["time"].append(current_time)
        euler_data["roll"].append(state["euler"][0])
        euler_data["pitch"].append(state["euler"][1])
        euler_data["yaw"].append(state["euler"][2])

        ang_vel_data["time"].append(current_time)
        ang_vel_data["x"].append(state["ang_vel"][0])
        ang_vel_data["y"].append(state["ang_vel"][1])
        ang_vel_data["z"].append(state["ang_vel"][2])

        # 그래프 선 업데이트
        line_roll.set_data(euler_data["time"], euler_data["roll"])
        line_pitch.set_data(euler_data["time"], euler_data["pitch"])
        line_yaw.set_data(euler_data["time"], euler_data["yaw"])
        line_ang_x.set_data(ang_vel_data["time"], ang_vel_data["x"])
        line_ang_y.set_data(ang_vel_data["time"], ang_vel_data["y"])
        line_ang_z.set_data(ang_vel_data["time"], ang_vel_data["z"])

        for ax in [ax1, ax2]:
            ax.set_xlim(max(0, current_time - 10), current_time + 1)

        plt.pause(0.01)

    try:
        print("Reading IMU... Press Ctrl+C to stop and save.")
        while True:
            update_plot()

    except KeyboardInterrupt:
        print("\nStopping and saving data...")
    finally:
        # 1. IMU 연결 종료
        imu.close()

        # 2. 그래프 이미지 저장
        # 전체 데이터가 보이도록 x축 범위를 다시 조정
        if euler_data["time"]:
            ax1.set_xlim(0, euler_data["time"][-1])
            ax2.set_xlim(0, ang_vel_data["time"][-1])
        
        plot_filename = f"imu_plot_{int(time.time())}.png"
        plt.savefig(plot_filename)
        print(f"Graph saved as {plot_filename}")

        # 3. CSV 데이터 저장 (나중에 분석하기 위함)
        csv_filename = f"imu_data_{int(time.time())}.csv"
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time", "roll", "pitch", "yaw", "ang_vel_x", "ang_vel_y", "ang_vel_z"])
            for i in range(len(euler_data["time"])):
                writer.writerow([
                    euler_data["time"][i],
                    euler_data["roll"][i], euler_data["pitch"][i], euler_data["yaw"][i],
                    ang_vel_data["x"][i], ang_vel_data["y"][i], ang_vel_data["z"][i]
                ])
        print(f"Raw data saved as {csv_filename}")
        
        # plt.show() 대신 close를 호출하여 리소스를 정리합니다.
        plt.close()
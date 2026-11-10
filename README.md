# LARC 2025 — Colour Perception and IMU State Estimation for an Autonomous Robot

![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![DepthAI](https://img.shields.io/badge/DepthAI-OAK--D-8A2BE2)
![Docker](https://img.shields.io/badge/Docker-Raspberry%20Pi%204-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Software for the 2025 **IEEE Latin American Robotics Competition (LARC) Open Challenge** robot,
running on a Raspberry Pi 4 with a Luxonis OAK-D. It has two subsystems:

- **Vision** (`src/vision/`): HSV segmentation of red, yellow and blue game elements with
  morphological closing and area-filtered contours.
- **State estimation** (`src/imu/`): an **Extended Kalman Filter** that fuses the OAK-D IMU's
  400 Hz accelerometer and gyroscope with its rotation-vector quaternion. It removes gravity in the
  world frame, integrates angular rate, and estimates position, velocity and orientation, with live plots.

The predecessor perception stack (YOLOv7 + stereo 3D localization) is in [LARC](https://github.com/NoisyScope/LARC).

## Layout

| Path | Purpose |
|---|---|
| `src/main.py` | Colour-detection loop |
| `src/vision/camera_stream.py`, `color_detection.py` | Camera wrapper and HSV detector |
| `src/utils/constants.py` | HSV thresholds per colour |
| `src/imu/imu_processing.py` | EKF, quaternion→rotation, gravity removal, gyro integration |
| `src/imu/imu_main.py`, `imu_plotting.py` | DepthAI IMU pipeline and live plotting |
| `Dockerfile` | DepthAI runtime image for the Pi |
| `docs/docker-raspberry-pi.md` | Installing Docker on Raspberry Pi OS 64-bit |

## Running

```bash
pip install -r requirements.txt
python src/main.py               # colour detection (q to quit)
cd src/imu && python imu_main.py # IMU EKF with an OAK-D connected
```

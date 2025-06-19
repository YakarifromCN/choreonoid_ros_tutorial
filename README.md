# Choreonoid ROS Tutorial: Go2 Inference Controller with ROS Integration

This project, developed for the "Agent Systems" course at the Graduate School of Information Science and Technology, The University of Tokyo, demonstrates the integration of a pre-trained AI model with the Robot Operating System (ROS) within the Choreonoid simulator.

---

## Overview

This project runs a pre-trained walking policy for a **Go2** quadruped robot inside the Choreonoid simulator. Its primary feature is the fusion of two key concepts:

- **AI-Powered Control**: An `InferenceController` loads a pre-trained neural network model using **LibTorch** (the PyTorch C++ API). This model is responsible for generating the robot's base walking gait.
- **Real-Time ROS Interaction**: The controller subscribes to the `/cmd_vel` ROS topic to receive velocity commands from an external source (e.g., keyboard input).

This architecture allows the robot to maintain autonomous walking behavior while enabling real-time interactive control via standard ROS tools.

---

## How It Works

The core logic is handled by the `InferenceController`, a custom C++ `SimpleController` plugin for Choreonoid. The data flow is as follows:

1. **User Input**: `teleop_twist_keyboard` captures keyboard commands.
2. **ROS Topic**: A `geometry_msgs/Twist` message is published to `/cmd_vel`.
3. **Controller**: `InferenceController` subscribes to `/cmd_vel` and receives the velocity command.
4. **AI Inference**: The velocity and robot state are fed into the pre-trained LibTorch model.
5. **Robot Action**: The model outputs joint positions, which are converted to torques using a PD controller and applied to the robot.

---

## Project Structure
```
<YOUR_PROJECT_DIR>/
├── ros/
│ └── src/
│ └── choreonoid_ros_tutorial/
│ ├── CMakeLists.txt
│ ├── package.xml
│ ├── README.md
│ ├── launch/
│ │ ├── go2_inference_shorttrack.launch
│ │ └── go2_inference_athletics.launch
│ └── src/
│ ├── CMakeLists.txt
│ ├── InferenceController.cpp
│ └── RttTankController.cpp
└── genesis_ws/
├── ... (Genesis and LibTorch files)
```
---

## Prerequisites and Dependencies

### Software Dependencies

- Ubuntu 20.04 LTS
- ROS Noetic
- Choreonoid
- LibTorch (PyTorch C++ API)
- `xterm`: install via `sudo apt install xterm`

### ROS Package Dependencies (see `package.xml`)

- `catkin`
- `choreonoid`, `choreonoid_ros`
- `std_msgs`, `sensor_msgs`, `image_transport`
- `teleop_twist_keyboard` (runtime dependency)

### Asset Paths

Replace `<YOUR_PROJECT_DIR>` with your project directory name (e.g., `Agent_System`).

- **LibTorch Path**: `~/<YOUR_PROJECT_DIR>/genesis_ws/libtorch`
- **Trained Model Path**: `~/<YOUR_PROJECT_DIR>/genesis_ws/logs/go2-walking/`

Create a symbolic link to the model:

```bash
cd ~/<YOUR_PROJECT_DIR>/genesis_ws/logs/go2-walking/
ln -s your_chosen_model_directory/ inference_target
```
---
## Build Instructions

> Note: Replace `<YOUR_PROJECT_DIR>` with your chosen main directory name.

### Navigate to Your Catkin Workspace

```bash
cd ~/<YOUR_PROJECT_DIR>/ros/
```
### Install Dependencies (Recommended)

```bash
rosdep install --from-paths src -r -y -i
```

### Build with Catkin

You must provide the path to your LibTorch installation via a CMake argument:

```bash
catkin build choreonoid_ros_tutorial \
  --force-cmake \
  --cmake-args -DTorch_DIR="${HOME}/<YOUR_PROJECT_DIR>/genesis_ws/libtorch/share/cmake/Torch"
```

## Execution Instructions

### Source the Workspace

```bash
cd ~/<YOUR_PROJECT_DIR>/ros/
source devel/setup.bash
```
### Launch the Simulation

- For **Short Track**:

  ```bash
  roslaunch choreonoid_ros_tutorial go2_inference_shorttrack.launch
  ```

- For **Athletics**:

  ```bash
  roslaunch choreonoid_ros_tutorial go2_inference_athletics.launch
  ```
## Usage and Controls

After launching, two windows will appear: the Choreonoid simulation and an `xterm` terminal.

- Click on the keyboard input terminal to make it the active window.
- Use the following keys to control the Go2 robot's target velocity:

```
i : Move forward
, : Move backward
j : Turn left
l : Turn right
k : Stop all movement
```

Follow the other on-screen instructions in the keyboard terminal for more options.

## Simulation Logging

Choreonoid provides a powerful logging feature to record and play back entire simulations.

### Recording a Simulation

1. Open Project: Start Choreonoid and load your desired project file (e.g., via `roslaunch`).
2. Add Log Item: In the Choreonoid menu, go to `File > New > WorldLogFile`.
3. Set Parent: In the Items view, drag the newly created `WorldLogFile` item and drop it onto the `AISTSimulator` item.
4. Configure Log File:
   - Set the `Log file` field (e.g., `my_run`)
   - Set `Time-stamp suffix` to `True` (optional)
5. Run and Stop: Start simulation and stop it when finished.
6. Save Log: Right-click `WorldLogFile` > `Save project as log playback archive`.

### Playing Back a Simulation

- Choreonoid will automatically switch to playback mode.
- Press the main **Play** button to watch the replay.
- To view later, open the `.cnoid` project file in Choreonoid.

## Appendix: Environment Setup from Scratch

This guide details the steps to set up a complete development environment on Ubuntu 20.04.

### Step 0: Create Project Directory

```bash
mkdir ~/<YOUR_PROJECT_DIR>
```

### Step 1: Install Base System and ROS Noetic

```bash
sudo apt update
sudo apt install curl git
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-noetic-desktop
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt install python3-vcstool python3-catkin-tools python3-rosdep
sudo rosdep init
rosdep update
```

### Step 2: Set up Python 3.10 Virtual Environment for Genesis

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv -y
mkdir -p ~/<YOUR_PROJECT_DIR>/genesis_ws
cd ~/<YOUR_PROJECT_DIR>/genesis_ws
python3.10 -m venv genesis_env
source genesis_env/bin/activate
```

### Step 3: Install Genesis, PyTorch, and AI Dependencies

```bash
cd ~/<YOUR_PROJECT_DIR>/genesis_ws
git clone https://github.com/kindsenior/Genesis.git -b agent_system_lecture2025
cd Genesis
pip install -r requirements_Ubuntu20.04_agent-system_cpu.txt
# or for GPU:
# pip install -r requirements_Ubuntu20.04_agent-system_cuda.txt
```

### Step 4: Install LibTorch (PyTorch C++ API)

```bash
cd ~/<YOUR_PROJECT_DIR>/genesis_ws
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.7.0+cpu.zip
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:${HOME}/<YOUR_PROJECT_DIR>/genesis_ws/libtorch/lib" >> ~/.bashrc
source ~/.bashrc
```

### Step 5: Create Catkin Workspace and Build Project

```bash
mkdir -p ~/<YOUR_PROJECT_DIR>/ros/src
cd ~/<YOUR_PROJECT_DIR>/ros/src
git clone https://github.com/agent-system/lecture2025.git
vcs import --input lecture2025/.rosinstall
git clone https://github.com/choreonoid/choreonoid_ros.git
cd ~/<YOUR_PROJECT_DIR>/ros
rosdep install --from-paths src -r -i -y
cd src/choreonoid/misc/script/
./install-requisites-ubuntu-20.04.sh
echo "export CNOID_USE_GLSL=0" >> ~/.bashrc
source ~/.bashrc
cd ~/<YOUR_PROJECT_DIR>/ros
catkin build choreonoid_ros_tutorial \
  --force-cmake \
  --cmake-args -DTorch_DIR="${HOME}/<YOUR_PROJECT_DIR>/genesis_ws/libtorch/share/cmake/Torch"
```

After these steps, your environment is fully configured. You can now proceed with the **Execution Instructions**.


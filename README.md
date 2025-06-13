# Choreonoid ROS Tutorial: Go2 Inference Controller with ROS Integration

This is a project for the "Agent Systems" course at the Graduate School of Information Science and Technology, The University of Tokyo.

## Overview

This project runs a pre-trained quadruped robot ("Go2") walking policy within the Choreonoid simulator.

Its key feature is the integration of the **Inference Controller** (from Lecture 7), which uses a pre-trained AI model, with **ROS (Robot Operating System) remote control** (from Lecture 8). This allows the robot to maintain its autonomous walking behavior via the AI policy while enabling a user to interactively control its target velocity in real-time using keyboard commands sent through the `/cmd_vel` ROS topic.

## Dependencies & Prerequisites

To successfully build and run this project, the following environment is required.

1.  **ROS Noetic**: A standard ROS environment.
2.  **Choreonoid**: The version of Choreonoid set up within the course's ROS workspace.
3.  **LibTorch (PyTorch C++ API)**:
    * It is assumed to be located at the following path:
        `path/to/your/genesis_ws/libtorch`
    * e.g., `~/Agent_System/genesis_ws/libtorch`

4.  **Genesis & Trained Models**:
    * The `Genesis` project is required for the Inference Controller to load the AI model (`.pt` file) and configuration file (`.yaml` file).
    * This project assumes that trained model logs exist at:
        `path/to/your/genesis_ws/logs/go2-walking/`
    * e.g., `~/Agent_System/genesis_ws/logs/go2-walking/`
    * Furthermore, a symbolic link named `inference_target` must be correctly configured to point to the desired model directory for inference.
        ```bash
        # Example command to create the symbolic link
        cd path/to/your/genesis_ws/logs/go2-walking/
        ln -s sub20_com_fric0.2-1.8_kp18-30_kv0.7-1.2_rotI0.01-0.15_iter200/ inference_target
        ```
5.  **xterm**:
    * Required by the launch file to open a new terminal window. If not installed, please install it with the following command:
        ```bash
        sudo apt update
        sudo apt install xterm
        ```

## Build Instructions

1.  Navigate to the root of your ROS workspace.
    ```bash
    cd path/to/your/agent_system_ws/
    ```
    e.g., `cd ~/Agent_System/ros/agent_system_ws/`

2.  (Recommended) Use `rosdep` to install system dependencies listed in `package.xml`.
    ```bash
    rosdep install --from-paths . -r -y -i
    ```

3.  Compile the project using `catkin build`. You must pass the path to your LibTorch installation as a CMake argument.
    ```bash
    catkin build choreonoid_ros_tutorial --force-cmake --cmake-args -DTorch_DIR="path/to/your/genesis_ws/libtorch/share/cmake/Torch"
    ```
    e.g., `-DTorch_DIR="${HOME}/Agent_System/genesis_ws/libtorch/share/cmake/Torch"`

## Execution Instructions

1.  Open a new terminal and source the workspace's setup file.
    ```bash
    cd path/to/your/agent_system_ws/ && source devel/setup.bash
    ```
    e.g., `cd ~/Agent_System/ros/agent_system_ws/ && source devel/setup.bash`

2.  Use the `roslaunch` command to start the simulator and the keyboard control node simultaneously.
    ```bash
    roslaunch choreonoid_ros_tutorial go2_inference_shorttrack.launch
    ```

## Usage & Controls

1.  After running the `roslaunch` command, the Choreonoid window and a new terminal window for keyboard input will appear.
2.  **You must click on the new keyboard input terminal to make it the active window.**
3.  Use the following keys to control the Go2 robot:
    * `i` : Move forward
    * `,` : Move backward
    * `j` : Turn left
    * `l` : Turn right
    * `k` : Stop
    * Please follow the other key instructions displayed in the `teleop_twist_keyboard` terminal.

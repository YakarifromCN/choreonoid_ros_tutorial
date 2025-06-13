cmake_minimum_required(VERSION 3.5.0)
project(choreonoid_ros_tutorial)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 找到專案所需的所有依賴項
# find_package會設定許多有用的變數，例如 *_INCLUDE_DIRS 和 *_LIBRARIES
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  image_transport
  choreonoid
)

# 【修正1】: 在這裡加入對Torch(LibTorch)的查找
# 這一行會使用您在catkin build指令中傳入的 -DTorch_DIR 參數
find_package(Torch REQUIRED)

catkin_package(
  # 這裡可以加入執行時的依賴項
  # EXEC_DEPENDS teleop_twist_keyboard
  SKIP_CMAKE_CONFIG_GENERATION
  SKIP_PKG_CONFIG_GENERATION
)

# 【修正2】: 將Catkin和Torch的標頭檔路徑都加入到編譯器的搜索路徑中
# 這一行是解決 "torch/torch.h not found" 錯誤的關鍵
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Torch_INCLUDE_DIRS}
)

# 告訴CMake去處理src子目錄下的CMakeLists.txt
add_subdirectory(src)

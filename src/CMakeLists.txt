# 創建 RttTankController 控制器 (來自第8次課的範例)
# 假設您保留了這個檔案
# choreonoid_add_simple_controller(RttTankController RttTankController.cpp)
# target_link_libraries(RttTankController ${roscpp_LIBRARIES})

# 創建 InferenceController 控制器 (您的課題目標)
# 確保您的類別和檔案名稱與這裡一致
choreonoid_add_simple_controller(InferenceController InferenceController.cpp)

# 將InferenceController與所有它需要的函式庫連結起來
target_link_libraries(InferenceController
  # 用於ROS通訊
  ${roscpp_LIBRARIES}
  # 用於AI模型推論
  ${TORCH_LIBRARIES}
  ${CHOREONOID_BASE_LIBRARIES}
  )
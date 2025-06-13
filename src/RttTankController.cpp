/*
 * RttTankController.cpp
 * Rtt 是 Ros Tank Tutorial 的縮寫 
 * 這個控制器繼承自Choreonoid的SimpleController，並整合了ROS通訊功能。
 */

// 引入Choreonoid控制器必要的標頭檔
#include <cnoid/SimpleController> // 

// 引入ROS相關的標頭檔
#include <geometry_msgs/Twist.h>   // 用於接收速度指令的訊息類型 
#include <ros/node_handle.h>     // 用於和ROS Master通訊，建立節點 
#include <mutex>                 // 用於線程同步，防止資料競爭 

using namespace cnoid;

class RttTankController : public SimpleController {
    // === 成員變數定義 ===

    // ROS節點句柄，用於管理此控制器作為ROS節點的所有通訊。
    // 使用unique_ptr是為了自動管理記憶體，確保在物件銷毀時能正確釋放資源。
    std::unique_ptr<ros::NodeHandle> node;

    // ROS訂閱者物件，用於接收來自特定ROS主題的訊息。 
    ros::Subscriber subscriber;

    // 用於儲存從 /cmd_vel 主題接收到的最新速度指令。 
    geometry_msgs::Twist latest_command_velocity;

    // 互斥鎖(Mutex)，用於保護 latest_command_velocity 變數。
    // 因為ROS的回呼函式(callback)和主控制迴圈(control)在不同線程中運行，
    // 需要用鎖來避免它們同時存取該變數，造成資料毀損。 
    std::mutex command_velocity_mutex;

    // 指向機器人模型中左右履帶連結(Link)的指標。
    Link* trackL, *trackR; // 

    // 指向砲塔偏航(Y)和俯仰(P)關節的指標。
    Link* turretJoint[2]; // 
    double q_ref[2], q_prev[2]; // 用於砲塔位置控制的參考角度和前一時刻的角度 

    // 模擬的時間步長。
    double dt; // 

public:
    // 當控制器被載入到專案中時會被呼叫。 
    virtual bool configure(SimpleControllerConfig* config) override {
        // 實例化ROS節點句柄，讓這個控制器準備好與ROS網路通訊。 
        node.reset(new ros::NodeHandle);
        return true;
    }

    // 在模擬開始前一刻會被呼叫，用於初始化。 
    virtual bool initialize(SimpleControllerIO* io) override {
        // 獲取指向機器人本體(Body)的指標。 
        Body* body = io->body();
        // 獲取模擬的時間步長(time step)。 
        dt = io->timeStep();

        // 根據連結名稱獲取左右履帶的Link物件指標。 
        trackL = body->link("TRACK_L");
        trackR = body->link("TRACK_R");

        // 啟用左右履帶的輸出，這裡我們控制的是關節速度。 
        io->enableOutput(trackL, JointVelocity);
        io->enableOutput(trackR, JointVelocity);

        // 根據連結名稱獲取砲塔關節的Link物件指標。 
        turretJoint[0] = body->link("TURRET_Y"); // 偏航 Yaw
        turretJoint[1] = body->link("TURRET_P"); // 俯仰 Pitch

        // 初始化砲塔控制
        for (int i = 0; i < 2; ++i) {
            Link* joint = turretJoint[i];
            // 將參考角度(q_ref)和前一時刻角度(q_prev)都設為目前的初始角度。 
            q_ref[i] = q_prev[i] = joint->q();
            // 設定關節的驅動模式為力矩(Torque)控制。 
            joint->setActuationMode(JointTorque);
            // 啟用關節的IO，以便讀取角度並輸出力矩。 
            io->enableIO(joint);
        }

        // === 設定ROS訂閱者 ===
        // 訂閱名為 "cmd_vel" 的主題，佇列(queue)大小為1。 
        // 當收到訊息時，呼叫本類別的 command_velocity_callback 函式進行處理。 
        subscriber = node->subscribe("cmd_vel", 1, &RttTankController::command_velocity_callback, this);
        return true;
    }

    // ROS訊息的回呼函式(Callback Function)。
    // 每當 "cmd_vel" 主題有新訊息發佈時，這個函式就會被ROS非同步呼叫。 
    void command_velocity_callback(const geometry_msgs::Twist& msg) {
        // 使用 lock_guard 自動管理互斥鎖，確保線程安全。 
        // 當離開此函式作用域時，鎖會自動釋放。
        std::lock_guard<std::mutex> lock(command_velocity_mutex);
        // 將收到的訊息(msg)複製到成員變數中，供 control() 函式使用。 
        latest_command_velocity = msg;
    }

    // 主控制迴圈，在模擬的每一個時間步都會被呼叫。
    virtual bool control() override {
        geometry_msgs::Twist command_velocity;
        {
            // 使用互斥鎖保護對共享變數的讀取。
            // 這裡使用一個獨立的作用域 {} 來盡可能縮小鎖的範圍，提高效率。 
            std::lock_guard<std::mutex> lock(command_velocity_mutex);
            // 將儲存的最新指令複製到一個局部變數中。 
            command_velocity = latest_command_velocity;
        }

        // === 履帶速度控制 ===
        // 根據 Twist 訊息中的線速度(linear.x)和角速度(angular.z)來計算左右履帶的目標速度。 
        // 這裡的0.5和0.3是根據機器人運動學模型設定的轉換係數。
        trackL->dq_target() = 0.5 * command_velocity.linear.x - 0.3 * command_velocity.angular.z; // 
        trackR->dq_target() = 0.5 * command_velocity.linear.x + 0.3 * command_velocity.angular.z; // 

        // === 砲塔位置PD控制 ===
        static const double P = 200.0; // P gain (比例增益) 
        static const double D = 50.0;  // D gain (微分增益) 
        for (int i = 0; i < 2; ++i) {
            Link* joint = turretJoint[i];
            double q = joint->q();                             // 讀取目前關節角度 
            double dq = (q - q_prev[i]) / dt;                  // 計算目前關節角速度 
            double dq_ref = 0.0;                               // 目標角速度為0 
            // PD控制公式：u = P * (目標位置 - 目前位置) + D * (目標速度 - 目前速度)
            joint->u() = P * (q_ref[i] - q) + D * (dq_ref - dq); // 
            q_prev[i] = q;                                     // 更新前一時刻的角度 
        }

        return true;
    }

    // 當控制器停止時會被呼叫（例如模擬結束時）。 
    virtual void stop() override {
        // 關閉訂閱者，停止接收ROS訊息，釋放資源。 
        subscriber.shutdown();
    }
};

// 這是Choreonoid需要的工廠函式宏，
// 用於將這個類別註冊為一個可以被Choreonoid動態載入的SimpleController外掛。 
CNOID_IMPLEMENT_SIMPLE_CONTROLLER_FACTORY(RttTankController)
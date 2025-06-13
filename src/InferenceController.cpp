#include <cnoid/MessageView>
#include <cnoid/SimpleController>
#include <cnoid/ValueTree>
#include <cnoid/YAMLReader>

#include <torch/torch.h>
#include <torch/script.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>

// ROS対応：ROS通訊必要的標頭檔
#include <ros/node_handle.h>
#include <geometry_msgs/Twist.h>
#include <mutex>

using namespace cnoid;
namespace fs = std::filesystem;

class InferenceController1 : public SimpleController
{
    Body* ioBody;
    double dt;
    double inference_dt = 0.02; // genesis dt is 0.02 sec
    size_t inference_interval_steps;

    Vector3 global_gravity;
    VectorXd last_action;
    VectorXd default_dof_pos;
    VectorXd target_dof_pos;
    std::vector<std::string> motor_dof_names;

    torch::jit::script::Module model;

    // Config values
    double P_gain;
    double D_gain;
    int num_actions;
    double action_scale;
    double ang_vel_scale;
    double lin_vel_scale;
    double dof_pos_scale;
    double dof_vel_scale;
    Vector3 command_scale;

    // Command
    Vector3d command; // 這個變數將由ROS更新，而不是亂數

    // ROS対応：刪除或註解掉不再需要的亂數生成器
    // size_t resample_interval_steps;
    size_t step_count = 0;
    // std::mt19937 rng;
    // std::uniform_real_distribution<double> dist_lin_x;
    // std::uniform_real_distribution<double> dist_lin_y;
    // std::uniform_real_distribution<double> dist_ang;
    
    // ROS対応：新增ROS通訊相關的成員變數
    std::unique_ptr<ros::NodeHandle> node;
    ros::Subscriber subscriber;
    geometry_msgs::Twist latest_command_velocity; // 儲存從ROS收到的最新指令
    std::mutex command_velocity_mutex;            // 保護latest_command_velocity的互斥鎖

public:
    virtual bool initialize(SimpleControllerIO* io) override
    {
        dt = io->timeStep();
        ioBody = io->body();

        inference_interval_steps = static_cast<int>(std::round(inference_dt / dt));
        std::ostringstream oss;
        oss << "inference_interval_steps: " << inference_interval_steps;
        MessageView::instance()->putln(oss.str());

        global_gravity = Vector3(0.0, 0.0, -1.0);

        for(auto joint : ioBody->joints()) {
            joint->setActuationMode(JointTorque);
            io->enableOutput(joint, JointTorque);
            io->enableInput(joint, JointAngle | JointVelocity);
        }
        io->enableInput(ioBody->rootLink(), LinkPosition | LinkTwist);

        // command
        command = Vector3d(0.0, 0.0, 0.0);

        // find the cfgs file
        fs::path inference_target_path = fs::path(std::getenv("HOME")) / fs::path("genesis_ws/logs/go2-walking/inference_target");
        fs::path cfgs_path = inference_target_path / fs::path("cfgs.yaml");
        if (!fs::exists(cfgs_path)) {
            oss << cfgs_path << " is not found!!!";
            MessageView::instance()->putln(oss.str());
            return false;
        }

        YAMLReader reader;
        auto root = reader.loadDocument(cfgs_path)->toMapping();
        auto env_cfg = root->findMapping("env_cfg");
        auto obs_cfg = root->findMapping("obs_cfg");
        P_gain = env_cfg->get("kp", 20);
        D_gain = env_cfg->get("kd", 0.5);
        num_actions = env_cfg->get("num_actions", 1);
        last_action = VectorXd::Zero(num_actions);
        default_dof_pos = VectorXd::Zero(num_actions);
        auto dof_names = env_cfg->findListing("dof_names");
        motor_dof_names.clear();
        for(int i=0; i<dof_names->size(); ++i){
            motor_dof_names.push_back(dof_names->at(i)->toString());
        }
        auto default_angles = env_cfg->findMapping("default_joint_angles");
        for(int i=0; i<motor_dof_names.size(); ++i){
            std::string name = motor_dof_names[i];
            default_dof_pos[i] = default_angles->get(name, 0.0);
        }
        target_dof_pos = default_dof_pos;
        action_scale = env_cfg->get("action_scale", 1.0);
        ang_vel_scale = obs_cfg->findMapping("obs_scales")->get("ang_vel", 1.0);
        lin_vel_scale = obs_cfg->findMapping("obs_scales")->get("lin_vel", 1.0);
        dof_pos_scale = obs_cfg->findMapping("obs_scales")->get("dof_pos", 1.0);
        dof_vel_scale = obs_cfg->findMapping("obs_scales")->get("dof_vel", 1.0);
        command_scale[0] = lin_vel_scale;
        command_scale[1] = lin_vel_scale;
        command_scale[2] = ang_vel_scale;
        
        // ROS対応：刪除或註解掉不再需要的亂數初始化部分
        // rng.seed(std::random_device{}());
        // dist_lin_x = std::uniform_real_distribution<double>(lin_vel_x_range[0], lin_vel_x_range[1]);
        // dist_lin_y = std::uniform_real_distribution<double>(lin_vel_y_range[0], lin_vel_y_range[1]);
        // dist_ang = std::uniform_real_distribution<double>(ang_vel_range[0], ang_vel_range[1]);

        // --- 以下的Torch模型載入部分保持不變 ---
        fs::path model_path = inference_target_path / fs::path("policy_traced.pt");
        if (!fs::exists(model_path)) {
            // ... (省略錯誤處理) ...
            return false;
        }
        model = torch::jit::load(model_path, torch::kCPU);
        model.to(torch::kCPU);
        model.eval();

        // ROS対応：在initialize函數結尾，初始化ROS節點並設定訂閱者
        node.reset(new ros::NodeHandle);
        subscriber = node->subscribe("cmd_vel", 1, &InferenceController1::rosCommandCallback, this);

        return true;
    }

    // ROS対応：新增一個ROS回呼函式，用於接收來自鍵盤的指令
    void rosCommandCallback(const geometry_msgs::Twist& msg) {
        std::lock_guard<std::mutex> lock(command_velocity_mutex);
        latest_command_velocity = msg;
    }

    bool inference(VectorXd& target_dof_pos, const Vector3d& angular_velocity, const Vector3d& projected_gravity, const VectorXd& joint_pos, const VectorXd& joint_vel) {
        // --- 這個推論函數的內部邏輯完全保持不變 ---
        // 它接收 command 向量，並不知道這個 command 是來自亂數還是ROS
        try {
            std::vector<float> obs_vec;
            for(int i=0; i<3; ++i) obs_vec.push_back(angular_velocity[i] * ang_vel_scale);
            for(int i=0; i<3; ++i) obs_vec.push_back(projected_gravity[i]);
            for(int i=0; i<3; ++i) obs_vec.push_back(command[i] * command_scale[i]);
            for(int i=0; i<num_actions; ++i) obs_vec.push_back((joint_pos[i] - default_dof_pos[i]) * dof_pos_scale);
            for(int i=0; i<num_actions; ++i) obs_vec.push_back(joint_vel[i] * dof_vel_scale);
            for(int i=0; i<num_actions; ++i) obs_vec.push_back(last_action[i]);
            auto input = torch::from_blob(obs_vec.data(), {1, (long)obs_vec.size()}, torch::kFloat32).to(torch::kCPU);
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input);
            torch::Tensor output = model.forward(inputs).toTensor();
            auto output_cpu = output.to(torch::kCPU);
            auto output_acc = output_cpu.accessor<float, 2>();
            VectorXd action(num_actions);
            for(int i=0; i<num_actions; ++i){
                last_action[i] = output_acc[0][i];
                action[i] = last_action[i];
            }
            target_dof_pos = action * action_scale + default_dof_pos;
        }
        catch (const c10::Error& e) {
            std::cerr << "Inference error: " << e.what() << std::endl;
        }
        return true;
    }

    virtual bool control() override
    {
        // ROS対応：用從ROS接收到的指令，來更新給AI模型的command向量
        {
            std::lock_guard<std::mutex> lock(command_velocity_mutex);
            command[0] = latest_command_velocity.linear.x;
            command[1] = latest_command_velocity.linear.y;
            command[2] = latest_command_velocity.angular.z;
        }

        // ROS対応：刪除或註解掉原有的亂數指令生成部分
        /*
        if(step_count % resample_interval_steps == 0){
            command[0] = dist_lin_x(rng);
            command[1] = dist_lin_y(rng);
            command[2] = dist_ang(rng);
            std::cout << "command velocity:" << command.transpose() << std::endl;
        }
        */

        // --- 以下的狀態獲取、推論呼叫、PD控制部分保持不變 ---
        const auto rootLink = ioBody->rootLink();
        const Isometry3d root_coord = rootLink->T();
        Vector3 angular_velocity = root_coord.linear().transpose() * rootLink->w();
        Vector3 projected_gravity = root_coord.linear().transpose() * global_gravity;

        VectorXd joint_pos(num_actions), joint_vel(num_actions);
        for(int i=0; i<num_actions; ++i){
            auto joint = ioBody->joint(motor_dof_names[i]);
            joint_pos[i] = joint->q();
            joint_vel[i] = joint->dq();
        }

        if (step_count % inference_interval_steps == 0) {
            inference(target_dof_pos, angular_velocity, projected_gravity, joint_pos, joint_vel);
        }

        for(int i=0; i<num_actions; ++i) {
            auto joint = ioBody->joint(motor_dof_names[i]);
            double q = joint->q();
            double dq = joint->dq();
            double u = P_gain * (target_dof_pos[i] - q) + D_gain * (- dq);
            joint->u() = u;
        }

        ++step_count;

        return true;
    }
    
    // ROS対応：新增stop函數，在控制器停止時關閉ROS訂閱者
    virtual void stop() override {
        subscriber.shutdown();
    }
};

CNOID_IMPLEMENT_SIMPLE_CONTROLLER_FACTORY(InferenceController1)
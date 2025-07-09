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
    VectorXd last_action; // 存储上一次的动作，作为下一次推理的输入
    VectorXd action_to_execute; // *** 新增：存储当前需要执行的动作 ***
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
    Vector3d command; // 這個變數將由ROS更新

    size_t step_count = 0;
    
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

        command = Vector3d(0.0, 0.0, 0.0);

        // find the cfgs file
        fs::path inference_target_path = fs::path(std::getenv("HOME")) / "agentsystem/genesis_ws/logs/go2-walking/test13";
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
        P_gain = env_cfg->get("kp", 30);
        D_gain = env_cfg->get("kd", 1.2);
        num_actions = env_cfg->get("num_actions", 1);
        
        // *** 修改：初始化新增的动作向量 ***
        last_action = VectorXd::Zero(num_actions);
        action_to_execute = VectorXd::Zero(num_actions);

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
        
        fs::path model_path = inference_target_path / fs::path("policy_traced.pt");
        if (!fs::exists(model_path)) {
            return false;
        }
        model = torch::jit::load(model_path, torch::kCPU);
        model.to(torch::kCPU);
        model.eval();

        node.reset(new ros::NodeHandle);
        subscriber = node->subscribe("cmd_vel", 1, &InferenceController1::rosCommandCallback, this);

        return true;
    }

    void rosCommandCallback(const geometry_msgs::Twist& msg) {
        std::lock_guard<std::mutex> lock(command_velocity_mutex);
        latest_command_velocity = msg;
    }

    // *** 修改：inference函数不再直接计算target_dof_pos, 而是填充一个输出向量 ***
    bool inference(VectorXd& output_action, const Vector3d& angular_velocity, const Vector3d& projected_gravity, const VectorXd& joint_pos, const VectorXd& joint_vel) {
        try {
            std::vector<float> obs_vec;
            for(int i=0; i<3; ++i) obs_vec.push_back(angular_velocity[i] * ang_vel_scale);
            for(int i=0; i<3; ++i) obs_vec.push_back(projected_gravity[i]);
            for(int i=0; i<3; ++i) obs_vec.push_back(command[i] * command_scale[i]);
            for(int i=0; i<num_actions; ++i) obs_vec.push_back((joint_pos[i] - default_dof_pos[i]) * dof_pos_scale);
            for(int i=0; i<num_actions; ++i) obs_vec.push_back(joint_vel[i] * dof_vel_scale);
            for(int i=0; i<num_actions; ++i) obs_vec.push_back(last_action[i]); // 使用上一次的动作作为输入

            auto input = torch::from_blob(obs_vec.data(), {1, (long)obs_vec.size()}, torch::kFloat32).to(torch::kCPU);
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input);
            torch::Tensor output = model.forward(inputs).toTensor();
            auto output_cpu = output.to(torch::kCPU);
            auto output_acc = output_cpu.accessor<float, 2>();
            
            // 填充输出向量
            for(int i=0; i<num_actions; ++i){
                output_action[i] = output_acc[0][i];
            }
        }
        catch (const c10::Error& e) {
            std::cerr << "Inference error: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    virtual bool control() override
    {
        // 从ROS更新指令
        {
            std::lock_guard<std::mutex> lock(command_velocity_mutex);
            command[0] = latest_command_velocity.linear.x;
            command[1] = latest_command_velocity.linear.y;
            command[2] = latest_command_velocity.angular.z;
        }

        // *** 关键逻辑修改：实现动作延迟 ***

        // 1. 使用上一个推理周期计算出的动作 `action_to_execute` 来设定当前的PD目标
        target_dof_pos = action_to_execute * action_scale + default_dof_pos;

        // 2. 如果到了推理的时刻，则计算*下一个*要执行的动作
        if (step_count % inference_interval_steps == 0) {
            
            // a. 获取当前机器人状态
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
            
            // b. 调用推理函数，计算出新动作并存储在 `action_to_execute` 中，供未来的循环使用
            inference(action_to_execute, angular_velocity, projected_gravity, joint_pos, joint_vel);

            // c. 更新 `last_action`，使其成为下一次推理的输入
            last_action = action_to_execute;
        }

        // 3. 执行PD控制，驱动机器人朝向 `target_dof_pos`
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
    
    virtual void stop() override {
        subscriber.shutdown();
    }
};

CNOID_IMPLEMENT_SIMPLE_CONTROLLER_FACTORY(InferenceController1)
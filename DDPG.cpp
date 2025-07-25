#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <tuple>
#include <queue>
#include <mutex>
#include <condition_variable>

class ReplayBuffer {
private:
    std::queue<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, float, bool>> buffer;
    std::mutex mtx;
    std::condition_variable cv;
    size_t max_size;

public:
    explicit ReplayBuffer(size_t max_size) : max_size(max_size) {}

    void add(torch::Tensor state, torch::Tensor action, torch::Tensor next_state, torch::Tensor reward, torch::Tensor done) {
        std::lock_guard<std::mutex> lock(mtx);
        int64_t batch_size = state.size(0);

        std::cout << "ReplayBuffer::add state size: " << state.sizes() << std::endl;
        std::cout << "ReplayBuffer::add action size: " << action.sizes() << std::endl;
        std::cout << "ReplayBuffer::add next_state size: " << next_state.sizes() << std::endl;
        std::cout << "ReplayBuffer::add reward size: " << reward.sizes() << std::endl;
        std::cout << "ReplayBuffer::add done size: " << done.sizes() << std::endl;

        for (int64_t i = 0; i < batch_size; ++i) {
            if (buffer.size() >= max_size) buffer.pop();

            buffer.emplace(
                state[i].clone(),
                action[i].clone(),
                next_state[i].clone(),
                reward[i].item<float>(),
                done[i].item<bool>()
            );
        }
        cv.notify_one();
    }

    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, float, bool>> sample(size_t batch_size) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return buffer.size() >= batch_size; });

        std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, float, bool>> batch;
        for (size_t i = 0; i < batch_size; ++i) {
            batch.push_back(buffer.front());
            buffer.pop();
        }
        return batch;
    }
};

class ActorImpl : public torch::nn::Module {
public:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    int state_dim, action_dim;

    ActorImpl(int state_dim, int action_dim) : state_dim(state_dim), action_dim(action_dim) {
        int hidden_dim = std::pow(2, std::ceil(std::log2(state_dim)));
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, action_dim));
    }

    torch::Tensor forward(torch::Tensor state) {
        std::cout << "Actor input state: " << state.sizes() << std::endl;
        auto x = torch::relu(fc1(state));
        x = torch::relu(fc2(x));
        return torch::sigmoid(fc3(x));
    }
};
TORCH_MODULE(Actor);

class CriticImpl : public torch::nn::Module {
public:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    int state_dim, action_dim;

    CriticImpl(int state_dim, int action_dim) : state_dim(state_dim), action_dim(action_dim) {
        int hidden_dim = std::pow(2, std::ceil(std::log2(state_dim + action_dim)));
        fc1 = register_module("fc1", torch::nn::Linear(state_dim + action_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, 1));
    }

    torch::Tensor forward(torch::Tensor state, torch::Tensor action) {
        std::cout << "Critic state size: " << state.sizes() << std::endl;
        std::cout << "Critic action size: " << action.sizes() << std::endl;

        if (state.dim() == 1) state = state.unsqueeze(0);
        if (action.dim() == 1) action = action.unsqueeze(0);

        auto x = torch::cat({state, action}, 1);
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        return fc3(x);
    }
};
TORCH_MODULE(Critic);

template <typename ModuleType>
void soft_update(ModuleType& target, const ModuleType& source, double tau) {
    torch::NoGradGuard no_grad;

    auto target_params = target->named_parameters();
    auto source_params = source->named_parameters();

    for (const auto& item : source_params) {
        const std::string& name = item.key();
        const auto& src = item.value().detach();  // ensure detached from graph
        auto& tgt = target_params[name];

        // avoid in-place by reassigning the value
        tgt.set_data((1.0 - tau) * tgt.detach() + tau * src);
    }
}

template <typename ModuleType>
void copy_parameters(ModuleType& target, const ModuleType& source) {
    auto source_params = source->parameters();
    auto target_params = target->parameters();
    if (source_params.size() != target_params.size()) {
        throw std::runtime_error("Mismatch in number of parameters between source and target.");
    }
    for (size_t i = 0; i < source_params.size(); ++i) {
        target_params[i].detach().copy_(source_params[i].detach());
    }
}

class DDPG {
public:
    Actor actor, actor_target;
    Critic critic, critic_target;
    torch::optim::Adam actor_optimizer, critic_optimizer;
    double tau, gamma;
    std::mutex actor_mutex;

    DDPG(int state_dim, int action_dim, double actor_lr, double critic_lr, double tau, double gamma)
        : actor(Actor(state_dim, action_dim)),
          actor_target(Actor(state_dim, action_dim)),
          critic(Critic(state_dim, action_dim)),
          critic_target(Critic(state_dim, action_dim)),
          actor_optimizer(actor->parameters(), actor_lr),
          critic_optimizer(critic->parameters(), critic_lr),
          tau(tau), gamma(gamma) {
        copy_parameters(actor_target, actor);
        copy_parameters(critic_target, critic);
    }

    torch::Tensor select_action(torch::Tensor state) {
        std::lock_guard<std::mutex> lock(actor_mutex);
        actor->eval();
        if (state.dim() == 1) state = state.unsqueeze(0);
        auto action = actor->forward(state);
        actor->train();
        return action;
    }

    void update(std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, float, bool>> &batch) {
        for (auto &[state, action, next_state, reward, done] : batch) {
            if (state.dim() == 1) state = state.unsqueeze(0);
            if (action.dim() == 1) action = action.unsqueeze(0);

            auto target_action = actor_target->forward(next_state);
            auto target_q = critic_target->forward(next_state, target_action);
            auto reward_tensor = torch::tensor(reward).to(target_q.device()).to(target_q.dtype());
            auto done_tensor = torch::tensor(done ? 1.0 : 0.0).to(target_q.device()).to(target_q.dtype());
            auto y = reward_tensor + gamma * (1 - done_tensor) * target_q;

            auto q = critic->forward(state, action);
            auto critic_loss = torch::mse_loss(q, y.detach());

            critic_optimizer.zero_grad();
            critic_loss.backward();
            critic_optimizer.step();

            auto actor_loss = -critic->forward(state, actor->forward(state)).mean();

            actor_optimizer.zero_grad();
            actor_loss.backward();
            actor_optimizer.step();

            soft_update(actor_target, actor, tau);
            soft_update(critic_target, critic, tau);
        }
    }

private:
    template <typename ModuleType>
    void soft_update(ModuleType& target, const ModuleType& source, double tau) {
        torch::NoGradGuard no_grad;

        auto target_params = target->named_parameters();
        auto source_params = source->named_parameters();

        for (const auto& item : source_params) {
            const std::string& name = item.key();
            const auto& src = item.value();
            auto& tgt = target_params[name];

            auto updated = tgt * (1.0 - tau) + src * tau;
            tgt.copy_(updated);
        }
    }
};

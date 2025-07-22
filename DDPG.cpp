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

    int64_t batch_size = state.size(0);  // assumes dim 0 is batch

    //Checks for runtime error
    std::cout << "ReplayBuffer::add state size: " << state.sizes() << std::endl;
    std::cout << "ReplayBuffer::add action size: " << action.sizes() << std::endl;
    std::cout << "ReplayBuffer::add next_state size: " << next_state.sizes() << std::endl;
    std::cout << "ReplayBuffer::add reward size: " << reward.sizes() << std::endl;
    std::cout << "ReplayBuffer::add done size: " << done.sizes() << std::endl;

    for (int64_t i = 0; i < batch_size; ++i) {
        if (buffer.size() >= max_size) {
            buffer.pop();
        }

        // Slice individual elements from batched tensors
        /*torch::Tensor s = state[i];
        torch::Tensor a = action[i];
        torch::Tensor ns = next_state[i];*/
        torch::Tensor s = state[i].unsqueeze(0);
        torch::Tensor a = action[i].unsqueeze(0);
        torch::Tensor ns = next_state[i].unsqueeze(0);
        float r = reward[i].item<float>();
        bool d = done[i].item<bool>();

        buffer.emplace(s, a, ns, r, d);
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

// Actor Neural Network
class ActorImpl : public torch::nn::Module {
public:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    int state_dim, action_dim;

    ActorImpl(int state_dim, int action_dim)
        : state_dim(state_dim), action_dim(action_dim) {
        int hidden_dim = std::pow(2, std::ceil(std::log2(state_dim)));

        // Define the layers
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, action_dim));
    }

    torch::Tensor forward(torch::Tensor state) {
        std::cout << "Actor input state: " << state.sizes() << std::endl;
        auto x = torch::relu(fc1(state));
        x = torch::relu(fc2(x));
        return torch::sigmoid(fc3(x));  // Outputs actions in range [0, 1]
    }
};
TORCH_MODULE(Actor);

// Critic Neural Network
class CriticImpl : public torch::nn::Module {
public:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    int state_dim, action_dim;

    CriticImpl(int state_dim, int action_dim)
        : state_dim(state_dim), action_dim(action_dim) {
        int hidden_dim = std::pow(2, std::ceil(std::log2(state_dim + action_dim)));

        // Define the layers
        fc1 = register_module("fc1", torch::nn::Linear(state_dim + action_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, 1));
    }

    /*torch::Tensor forward(torch::Tensor state, torch::Tensor action) {
        std::cout << "Critic state: " << state.sizes() << std::endl;
        std::cout << "Critic action: " << action.sizes() << std::endl;
        auto x = torch::cat({state, action}, 1);
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        return fc3(x);  // Q-value output
    }*/
     torch::Tensor forward(torch::Tensor state, torch::Tensor action) {
        std::cout << "Critic state size: " << state.sizes() << std::endl;
        std::cout << "Critic action size: " << action.sizes() << std::endl;

        // Defensive: unsqueeze if 1D
        if (state.dim() == 1) {
            state = state.unsqueeze(0);
        }
        if (action.dim() == 1) {
            action = action.unsqueeze(0);
        }

        auto x = torch::cat({state, action}, 1);
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        return fc3(x);
    }
};
TORCH_MODULE(Critic);

// Helper function for copying parameters
template <typename ModuleType>
void copy_parameters(ModuleType& target, const ModuleType& source) {
    auto source_params = source->parameters();
    auto target_params = target->parameters();

    if (source_params.size() != target_params.size()) {
        throw std::runtime_error("Mismatch in number of parameters between source and target.");
    }

    for (size_t i = 0; i < source_params.size(); ++i) {
        // Detach to avoid in-place ops on leaf variables requiring grad
        target_params[i].detach().copy_(source_params[i].detach());
    }
}

// DDPG Agent
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
        // Initialize target networks
        copy_parameters(actor_target, actor);
        copy_parameters(critic_target, critic);
    }

    /*torch::Tensor select_action(torch::Tensor state) {
    actor->eval();
    if (state.dim() == 1) {
        state = state.unsqueeze(0);  // Convert [features] to [1, features]
    }
    auto action = actor->forward(state);
    actor->train();
    return action.squeeze(0);  // Return to 1D tensor if needed
    }*/
    /*torch::Tensor select_action(torch::Tensor state) {
    actor->eval();
    if (state.dim() == 1) {
        state = state.unsqueeze(0);  // [1, state_dim]
    }
    auto action = actor->forward(state);
    actor->train();
    return action;  // ← REMOVE .squeeze(0)
    }*/
   torch::Tensor select_action(torch::Tensor state) {
    std::lock_guard<std::mutex> lock(actor_mutex);  // Prevent concurrent access
    actor->eval();
    if (state.dim() == 1) {
        state = state.unsqueeze(0);
    }
    auto action = actor->forward(state);
    actor->train();
    return action;
    }

    void update(std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, float, bool>> &batch) {
        for (auto &[state, action, next_state, reward, done] : batch) {
            // For runtime error
            if (state.dim() == 1) {
                state = state.unsqueeze(0);
            }
            if (action.dim() == 1) {
                action = action.unsqueeze(0);
            }

            // Target Q-value
            auto target_action = actor_target->forward(next_state);
            auto target_q = critic_target->forward(next_state, target_action);
            auto y = reward + gamma * (1 - done) * target_q;

            // Current Q-value
            auto q = critic->forward(state, action);

            // Critic loss
            auto critic_loss = torch::mse_loss(q, y.detach());

            // Optimize critic
            critic_optimizer.zero_grad();
            critic_loss.backward();
            critic_optimizer.step();

            // Actor loss
            auto actor_loss = -critic->forward(state, actor->forward(state)).mean();

            // Optimize actor
            actor_optimizer.zero_grad();
            actor_loss.backward();
            actor_optimizer.step();

            // Soft update target networks
            soft_update(actor_target, actor, tau);
            soft_update(critic_target, critic, tau);
        }
    }

private:
    template <typename ModuleType>
    void soft_update(ModuleType& target, const ModuleType& source, double tau) {
        for (size_t i = 0; i < target->parameters().size(); ++i) {
            // Detach all tensors before operations to avoid in-place errors
            auto updated_param = tau * source->parameters()[i].detach() +
                                 (1 - tau) * target->parameters()[i].detach();
            target->parameters()[i].detach().copy_(updated_param);
        }
    }
};

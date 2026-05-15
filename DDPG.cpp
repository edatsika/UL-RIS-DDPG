#pragma once
#include <torch/torch.h>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <random>
#include <atomic>

struct Experience {
    torch::Tensor state;
    torch::Tensor action;
    torch::Tensor next_state;
    float reward{};
    bool done{};
};

struct TrainingMetrics {
    float actor_loss;
    float critic_loss;
    float avg_q_value;      
    float avg_reward;       
};

class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t max_size)
        : max_size_(max_size), buffer_(max_size), write_pos_(0), 
          current_size_(0), rng_(std::random_device{}()) {}

    void add(torch::Tensor state, torch::Tensor action, torch::Tensor next_state, float reward, bool done) {
        // Clone out of lock
        Experience exp{state.clone(), action.clone(), next_state.clone(), reward, done};
        size_t slot;
        {
            std::lock_guard<std::mutex> lk(write_mtx_);
            slot = write_pos_ % max_size_;
            ++write_pos_;
            if (current_size_ < max_size_) ++current_size_;
            total_collected_.fetch_add(1, std::memory_order_relaxed);
            reward_sum_ += reward; 
        }
        {
            std::lock_guard<std::mutex> lk(stripe_mtx_[slot % 16]);
            buffer_[slot] = std::move(exp);
        }
        cv_.notify_all(); 
    }

    // Use stop flag to avoid deadlocks
    std::vector<Experience> sample(size_t batch_size, std::atomic<bool>& stop_flag) {
        size_t cur;
        {
            std::unique_lock<std::mutex> lk(write_mtx_);
            cv_.wait(lk, [&]{ return current_size_ >= batch_size || stop_flag.load(std::memory_order_relaxed); });
            
            if (stop_flag.load(std::memory_order_relaxed) && current_size_ < batch_size) {
                return {}; 
            }
            cur = current_size_;
        }
        
        std::vector<size_t> idx(batch_size);
        {
            std::lock_guard<std::mutex> lk(rng_mtx_);
            std::uniform_int_distribution<size_t> dist(0, cur - 1);
            for (auto& i : idx) i = dist(rng_);
        }
        
        std::vector<Experience> batch;
        batch.reserve(batch_size);
        for (size_t i : idx) {
            std::lock_guard<std::mutex> lk(stripe_mtx_[i % 16]);
            // Clone when reading from buffer
            batch.push_back({
                buffer_[i].state.clone(),
                buffer_[i].action.clone(),
                buffer_[i].next_state.clone(),
                buffer_[i].reward,
                buffer_[i].done
            });
        }
        return batch;
    }

    size_t size() const { 
        std::lock_guard<std::mutex> lk(write_mtx_); 
        return current_size_; 
    }
    long get_total_collected() const { 
        return total_collected_.load(std::memory_order_relaxed); 
    }
    float get_avg_reward() const {
        std::lock_guard<std::mutex> lk(write_mtx_);
        long total = total_collected_.load(std::memory_order_relaxed);
        return (total > 0) ? (reward_sum_ / total) : 0.0f;
    }
    void notify_stop() {
    cv_.notify_all(); // wake up sleeping sample
    }

private:
    size_t max_size_;
    std::vector<Experience> buffer_;
    size_t write_pos_;
    size_t current_size_;
    std::atomic<long> total_collected_{0}; 
    float reward_sum_{0.0f};  
    mutable std::mutex write_mtx_;
    mutable std::mutex stripe_mtx_[16];
    std::mutex rng_mtx_;
    std::condition_variable cv_;
    std::mt19937 rng_;
};

class OUNoise {
public:
    OUNoise(int size, float mu = 0.0f, float theta = 0.15f, float sigma = 0.2f)
        : size_(size), mu_(mu), theta_(theta), sigma_(sigma),
          state_(size, mu), rng_(std::random_device{}()) {}

    std::vector<float> sample() {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < size_; ++i) {
            float dx = theta_ * (mu_ - state_[i]) + sigma_ * dist(rng_);
            state_[i] += dx;
        }
        return state_;
    }
    void reset() {
        std::fill(state_.begin(), state_.end(), mu_);
    }
private:
    int size_;
    float mu_, theta_, sigma_;
    std::vector<float> state_;
    std::mt19937 rng_;
};

class ActorImpl : public torch::nn::Module {
public:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    ActorImpl(int state_dim, int action_dim) {
        int hidden = 256; 
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, hidden));
        fc2 = register_module("fc2", torch::nn::Linear(hidden, hidden));
        fc3 = register_module("fc3", torch::nn::Linear(hidden, action_dim));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        return torch::sigmoid(fc3(x));
    }
};
TORCH_MODULE(Actor);

class CriticImpl : public torch::nn::Module {
public:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    CriticImpl(int state_dim, int action_dim) {
        int in_dim = state_dim + action_dim;
        int hidden = 256;
        fc1 = register_module("fc1", torch::nn::Linear(in_dim, hidden));
        fc2 = register_module("fc2", torch::nn::Linear(hidden, hidden));
        fc3 = register_module("fc3", torch::nn::Linear(hidden, 1));
    }
    torch::Tensor forward(torch::Tensor state, torch::Tensor action) {
        auto x = torch::cat({state, action}, 1);
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        return fc3(x);
    }
};
TORCH_MODULE(Critic);

class DDPG {
public:
    DDPG(int state_dim, int action_dim, double actor_lr, double critic_lr, 
         double tau, double gamma, bool use_noise = true)
        : actor_(Actor(state_dim, action_dim)), 
          actor_target_(Actor(state_dim, action_dim)),
          critic_(Critic(state_dim, action_dim)), 
          critic_target_(Critic(state_dim, action_dim)),
          actor_opt_(actor_->parameters(), actor_lr), 
          critic_opt_(critic_->parameters(), critic_lr),
          tau_(tau), gamma_(gamma) {
        
        torch::NoGradGuard ng;
        auto sp = actor_->parameters(); auto tp = actor_target_->parameters();
        for (size_t i = 0; i < sp.size(); ++i) tp[i].copy_(sp[i]);
        
        auto sc = critic_->parameters(); auto tc = critic_target_->parameters();
        for (size_t i = 0; i < sc.size(); ++i) tc[i].copy_(sc[i]);
    }

    torch::Tensor select_action(const torch::Tensor& state, const torch::Tensor* noise_batch = nullptr) {
        std::shared_lock<std::shared_mutex> lk(agent_mutex_); // Lock for agent
        torch::NoGradGuard ng;
        auto s = (state.dim() == 1) ? state.unsqueeze(0) : state;
        auto action = actor_->forward(s).detach();

        if (noise_batch) {
            // noise_batch already [B, action_dim]: different noise per env
            action = torch::clamp(action + (*noise_batch) * 0.1f, 0.0f, 1.0f);
        }
        /*if (use_noise_ && add_noise) {
            auto n_vector = noise_.sample();
            auto noise_tensor = torch::tensor(n_vector, torch::kFloat).unsqueeze(0).expand_as(action);
            action = torch::clamp(action + noise_tensor * 0.1f, 0.0f, 1.0f);
        }*/
        return action;
    }

    TrainingMetrics update(std::vector<Experience>& batch) {
        // lock update as a whole to avoid actor-critic race
        std::unique_lock<std::shared_mutex> lk(agent_mutex_);

        std::vector<torch::Tensor> sv, av, nsv;
        std::vector<float> rv, dv;
        for (auto& exp : batch) {
            sv.push_back(exp.state.dim() == 1 ? exp.state.unsqueeze(0) : exp.state);
            av.push_back(exp.action.dim() == 1 ? exp.action.unsqueeze(0) : exp.action);
            nsv.push_back(exp.next_state.dim() == 1 ? exp.next_state.unsqueeze(0) : exp.next_state);
            rv.push_back(exp.reward);
            dv.push_back(exp.done ? 1.0f : 0.0f);
        }

        auto s = torch::cat(sv, 0); 
        auto a = torch::cat(av, 0); 
        auto ns = torch::cat(nsv, 0);
        auto r = torch::tensor(rv).unsqueeze(1).to(s.dtype());
        auto d = torch::tensor(dv).unsqueeze(1).to(s.dtype());

        float c_loss_val, a_loss_val, avg_q;

        // Critic update
        {
            auto tgt_a = actor_target_->forward(ns);
            auto tgt_q = critic_target_->forward(ns, tgt_a);
            auto y = r + gamma_ * (1.0f - d) * tgt_q;
            auto q = critic_->forward(s, a);

            avg_q = q.mean().item<float>();  

            //auto critic_loss = torch::mse_loss(q, y.detach());
            auto critic_loss = torch::smooth_l1_loss(q, y.detach());
            // smooth_l1 = Huber with delta=1: L2 for small errors, L1 for big errors, to avoid spikes
            c_loss_val = critic_loss.template item<float>();
            
            critic_opt_.zero_grad(); 
            critic_loss.backward(); 

            for (auto& param : critic_->parameters()) {
                if (param.grad().defined()) {
                    param.grad().clamp_(-1.0, 1.0); 
                }
            }
            critic_opt_.step();
        }

        // Actor update
        {
            auto actor_loss = -critic_->forward(s, actor_->forward(s)).mean();
            a_loss_val = actor_loss.template item<float>();
            
            actor_opt_.zero_grad(); 
            actor_loss.backward(); 

            for (auto& param : actor_->parameters()) {
                if (param.grad().defined()) {
                    param.grad().clamp_(-1.0, 1.0); 
                }
            }
            actor_opt_.step();

            // Target networks soft update
            torch::NoGradGuard ng;
            auto update_fn = [&](auto& target, auto& source) {
                auto tgt = target->named_parameters();
                for (const auto& kv : source->named_parameters())
                    tgt[kv.key()].set_data((1.0 - tau_) * tgt[kv.key()] + tau_ * kv.value());
            };
            update_fn(actor_target_, actor_);
            update_fn(critic_target_, critic_);
        }
        
        float avg_reward = 0.0f;
        for (float rew : rv) avg_reward += rew;
        avg_reward /= rv.size();
        
        return {a_loss_val, c_loss_val, avg_q, avg_reward};
    }

private:
    Actor actor_, actor_target_;
    Critic critic_, critic_target_;
    torch::optim::Adam actor_opt_, critic_opt_;
    double tau_, gamma_;
    std::shared_mutex agent_mutex_; 
};

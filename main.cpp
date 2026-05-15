#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <iomanip>
#include "DDPG.cpp"
#include "env.cpp"

//  collect_worker runs until stop_flag is set by the trainer, producers are always ready to supply fresh data
//  The trainer (consumer) decides when it has learned enough and signals shutdown.

void collect_worker(int                thread_id,
                    int                num_users,
                    int                num_ris,
                    int                num_elements,
                    double             bandwidth,
                    int                vec_size,
                    int                max_steps,
                    int                action_dim,
                    DDPG&              agent,
                    ReplayBuffer&      buffer,
                    std::atomic<bool>& stop_flag,
                    std::atomic<float>& episode_reward_sum,
                    std::atomic<int>&   episode_count)
{
    Environment env(num_users, num_ris, num_elements, bandwidth, vec_size);
    int episodes = 0;

    // 1 OUNoise per env 
    std::vector<OUNoise> noises;
    noises.reserve(vec_size);
    for (int b = 0; b < vec_size; ++b)
        noises.emplace_back(action_dim); 

    while (!stop_flag.load(std::memory_order_relaxed)) {
        torch::Tensor state = env.reset();

        // Reset each OU process at beginning of episode
        for (auto& n : noises) n.reset();

        float ep_reward = 0.0f;

        for (int step = 0; step < max_steps; ++step) {
            if (stop_flag.load(std::memory_order_relaxed)) break;

            // Derive [B, action_dim] noise tensor, different per env
            std::vector<torch::Tensor> noise_rows;
            noise_rows.reserve(vec_size);
            for (int b = 0; b < vec_size; ++b) {
                auto n_vec = noises[b].sample();   // [action_dim]
                noise_rows.push_back(
                    torch::tensor(n_vec, torch::kFloat).unsqueeze(0)  // [1, action_dim]
                );
            }
            auto noise_batch = torch::cat(noise_rows, 0);  // [B, action_dim]

            torch::Tensor action = agent.select_action(state, &noise_batch);  // with noise
            auto [next_state, reward, done] = env.step(action);

            for (int b = 0; b < vec_size; ++b) {
                float r = reward[b].item<float>();
                // Scale reward to reasonable range
                float scaled_r = r * 10.0f;  // Make rewards more significant
                
                buffer.add(
                    state     [b],
                    action    [b],
                    next_state[b],
                    scaled_r,  // Use scaled reward
                    done      [b].item<bool>()
                );
                ep_reward += r;  // Track original reward
            }
            state = next_state;
        }
        
        ++episodes;
        //episode_reward_sum.fetch_add(ep_reward / vec_size, std::memory_order_relaxed);
        // no fetch_add for floats?
        {
            float current_sum = episode_reward_sum.load(std::memory_order_relaxed);
            float update_val = ep_reward / vec_size;
            while (!episode_reward_sum.compare_exchange_weak(current_sum, current_sum + update_val,
                                                     std::memory_order_relaxed,
                                                     std::memory_order_relaxed));
        }
        episode_count.fetch_add(1, std::memory_order_relaxed);
    }

    std::cout << "[Collector " << thread_id << "] stopped after "
              << episodes << " episodes.\n";   

}

// train_worker: wontrols the lifetime of the whole run, sets stop_flag when training is complete, 
// which causes all collectors to exit on their next loop-top check.

void train_worker(DDPG&              agent,
                  ReplayBuffer&      buffer,
                  int                batch_size,
                  int                iterations,
                  int                log_every,
                  std::atomic<bool>& stop_flag,
                  std::atomic<float>& episode_reward_sum,
                  std::atomic<int>&   episode_count)
{
    std::cout << "[Trainer] Waiting for initial data (batch_size=" << batch_size << ")..." << std::endl;

    for (int i = 0; i < iterations && !stop_flag.load(std::memory_order_relaxed); ++i) {
       //auto batch = buffer.sample(static_cast<size_t>(batch_size));
       auto batch = buffer.sample(static_cast<size_t>(batch_size), stop_flag);
        if (batch.empty()) break;
        TrainingMetrics metrics = agent.update(batch);

        if (i % log_every == 0) {
            int ep_count = episode_count.load(std::memory_order_relaxed);
            float avg_ep_reward = (ep_count > 0) 
                ? episode_reward_sum.load(std::memory_order_relaxed) / ep_count 
                : 0.0f;
            
            std::cout << "[Trainer] Step " << std::setw(4) << i 
                      << " Episodes: " << std::setw(4) << ep_count
                      << " Avg Ep Reward: " << std::fixed << std::setprecision(4) << avg_ep_reward
                      << " A-Loss: " << std::setw(8) << metrics.actor_loss
                      << " C-Loss: " << std::setw(8) << metrics.critic_loss
                      << " Avg Q: " << std::setw(8) << metrics.avg_q_value
                      << " Batch Reward: " << metrics.avg_reward << std::endl;
        }
    }

    stop_flag.store(true, std::memory_order_relaxed);
    buffer.notify_stop();
    std::cout << "[Trainer] Finished training." << std::endl;
}


int main() {
    constexpr int    num_users    = 5;
    constexpr int    num_ris      = 2;
    constexpr int    num_elements = 32;
    constexpr double bandwidth    = 1e6;

    constexpr int num_collectors  = 3;    // producer threads
    constexpr int vec_size        = 4;    // sub-environments per collector
    constexpr int max_steps       = 50;   // steps per episode

    constexpr int    train_iters  = 10000;
    constexpr int    batch_size   = 256;
    constexpr int    log_every    = 50;
    constexpr size_t buf_capacity = 50000;

    Environment  probe(num_users, num_ris, num_elements, bandwidth);
    DDPG         agent(probe.state_dim, probe.action_dim,
                       5e-6, 1e-4, 5e-4, 0.99); // 1e-3, 1e-3, 5e-3, 0.99); slower actor
    ReplayBuffer buffer(buf_capacity);
    std::atomic<bool> stop_flag{false};
    std::atomic<float> episode_reward_sum{0.0f};
    std::atomic<int> episode_count{0};

    std::cout << "state_dim="   << probe.state_dim
              << "  action_dim=" << probe.action_dim << "\n"
              << num_collectors << " collectors × vec_size=" << vec_size
              << "  +  1 trainer\n";

    // Collectors run until trainer sets stop_flag
    std::vector<std::thread> collectors;
    collectors.reserve(num_collectors);
    for (int t = 0; t < num_collectors; ++t)
        collectors.emplace_back(collect_worker,
                                t,
                                num_users, num_ris, num_elements, bandwidth,
                                vec_size, max_steps,
                                probe.action_dim,
                                std::ref(agent),
                                std::ref(buffer),
                                std::ref(stop_flag),
                                std::ref(episode_reward_sum),
                                std::ref(episode_count));

    // Trainer sets stop_flag when done
    std::thread trainer(train_worker,
                        std::ref(agent), std::ref(buffer),
                        batch_size, train_iters, log_every,
                        std::ref(stop_flag),
                        std::ref(episode_reward_sum),
                        std::ref(episode_count));


    trainer.join();                         // wait for trainer first
    for (auto& t : collectors) t.join();   // collectors exit via stop_flag

    std::cout << "Buffer size: " << buffer.size() << "\n";
    std::cout << "Total episodes: " << episode_count.load() << "\n";
    std::cout << "Buffer size: " << buffer.size() << "\n";
    std::cout << "Final avg episode reward: " << (episode_reward_sum.load() / episode_count.load()) << "\n";
    return 0;
}

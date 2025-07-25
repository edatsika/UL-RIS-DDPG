#include <iostream>
#include <omp.h>
#include "DDPG.cpp"
#include "env.cpp"

void collect_experience(Environment& env, DDPG& ddpg, ReplayBuffer& shared_buffer, int episodes, int steps_per_episode) {
    for (int ep = 0; ep < episodes; ++ep) {
        torch::Tensor state = env.reset();
        for (int step = 0; step < steps_per_episode; ++step) {
            torch::Tensor action = ddpg.select_action(state);
            auto [next_state, reward, done] = env.step(action);
            shared_buffer.add(state, action, next_state, reward, done);
            state = next_state;
            if (done.item<bool>()) break;
        }
    }
    std::cout << "Thread " << omp_get_thread_num() << " finished collecting." << std::endl;
}

void train_agent(DDPG &agent, ReplayBuffer &replay_buffer, int batch_size, int iterations) {
    std::cout << "Starting training..." << std::endl;
    for (int i = 0; i < iterations; ++i) {
        auto batch = replay_buffer.sample(batch_size);
        agent.update(batch);
    }
}

int main() {
    const int num_users = 5, num_ris = 2, num_elements = 32;
    const double bandwidth = 1e6;
    const int num_threads = 4;
    const int episodes_per_thread = 5;
    const int max_steps = 10;

    Environment dummy_env(num_users, num_ris, num_elements, bandwidth);
    DDPG agent(dummy_env.state_dim, dummy_env.action_dim, 0.001, 0.001, 0.005, 0.99);
    ReplayBuffer shared_buffer(1000);

    // Parallel data collection using OpenMP
    /*#pragma omp parallel num_threads(num_threads)
    {
        Environment local_env(num_users, num_ris, num_elements, bandwidth);
        collect_experience(local_env, agent, shared_buffer, episodes_per_thread, max_steps);
    }*/
   #pragma omp parallel for
    for (int i = 0; i < num_threads; ++i) {
        Environment local_env(num_users, num_ris, num_elements, bandwidth);
        collect_experience(local_env, agent, shared_buffer, episodes_per_thread, max_steps);
    }

    train_agent(agent, shared_buffer, 8, 100); //64, 100
    std::cout << "Training completed with OpenMP multithreading." << std::endl;
    return 0;
}

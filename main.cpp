#include <iostream>
#include "DDPG.cpp"
#include "env.cpp"

void collect_experience(Environment& env, DDPG& ddpg, ReplayBuffer& replay_buffer, int episodes, int steps_per_episode) {
    for (int ep = 0; ep < episodes; ++ep) {
        torch::Tensor state = env.reset();

        for (int step = 0; step < steps_per_episode; ++step) {
            torch::Tensor action = ddpg.select_action(state);
            auto [next_state, reward, done] = env.step(action);
            replay_buffer.add(state, action, next_state, reward, done);
            state = next_state;
            if (done.item<bool>()) break;
        }
    }
}

void train_agent(DDPG &agent, ReplayBuffer &replay_buffer, int batch_size, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        auto batch = replay_buffer.sample(batch_size);
        agent.update(batch);
    }
}

int main() {
    int num_users = 5, num_ris = 2, num_elements = 32;
    double bandwidth = 1e6;  // 1 MHz
    Environment env(num_users, num_ris, num_elements, bandwidth);
    DDPG agent(env.state_dim, env.action_dim, 0.001, 0.001, 0.005, 0.99);
    ReplayBuffer replay_buffer(1000);

    const int episodes = 40;       // total episodes (was 4 threads * 10 each)
    const int max_steps = 200;

    // Single-threaded data collection
    collect_experience(env, agent, replay_buffer, episodes, max_steps);

    // Train the agent
    train_agent(agent, replay_buffer, 64, 100);

    std::cout << "Training completed (single-threaded data collection)." << std::endl;
    return 0;
}

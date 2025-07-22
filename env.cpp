#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <cmath>

class Environment {
public:
    int num_users, num_ris, num_elements;
    int state_dim, action_dim;
    double bandwidth;

    // Complex-valued tensors represented with real and imag parts
    torch::Tensor g_km_real, g_km_imag;
    torch::Tensor h_km_real, h_km_imag;
    torch::Tensor theta_kmn_real, theta_kmn_imag;

    torch::Tensor snr_km;      // [B, K, M]
    torch::Tensor u_km;        // [B, K, M]
    torch::Tensor tau_sched;   // [B, K]
    torch::Tensor rho_k;       // [B, K]

    Environment(int num_users_, int num_ris_, int num_elements_, double bandwidth_)
        : num_users(num_users_), num_ris(num_ris_), num_elements(num_elements_), bandwidth(bandwidth_) {
        
        state_dim = num_users * num_ris   // snr_km
                  + num_users * num_ris   // u_km
                  + num_users             // tau_sched
                  + num_users;            // rho_k

        action_dim = 2 * num_users                          // rho_k + tau_sched
                   + 2 * num_users * num_ris * num_elements // theta_kmn (real + imag)
                   + num_users * num_ris;                   // u_km
    }

    // 	Initialized real & imag parts of g_km, h_km, theta_kmn
    torch::Tensor reset() {
        int batch_size = 1;

        g_km_real = torch::rand({batch_size, num_users, num_ris, num_elements});
        g_km_imag = torch::rand({batch_size, num_users, num_ris, num_elements});
        h_km_real = torch::rand({batch_size, num_users, num_ris, num_elements});
        h_km_imag = torch::rand({batch_size, num_users, num_ris, num_elements});

        snr_km = torch::zeros({batch_size, num_users, num_ris});
        u_km = torch::zeros({batch_size, num_users, num_ris});
        tau_sched = torch::rand({batch_size, num_users});
        rho_k = torch::rand({batch_size, num_users});
        theta_kmn_real = torch::rand({batch_size, num_users, num_ris, num_elements});
        theta_kmn_imag = torch::rand({batch_size, num_users, num_ris, num_elements});

        for (int k = 0; k < num_users; ++k) {
            int random_ris = std::rand() % num_ris;
            u_km[0][k][random_ris] = 1;
        }

        return get_state();
    }

    torch::Tensor get_state() {
        auto snr_flat = snr_km.flatten(1);
        auto u_flat = u_km.flatten(1);
        return torch::cat({snr_flat, u_flat, tau_sched, rho_k}, 1);
    }

    // Parses and reshapes real/imag parts from action tensor
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> step(torch::Tensor action) {
        int batch_size = action.size(0);

        auto rho_k_ = action.slice(1, 0, num_users).clone();  // [B, K]
        auto tau_sched_ = action.slice(1, num_users, 2 * num_users).clone();  // [B, K]

        int theta_start = 2 * num_users;
        int theta_len = 2 * num_users * num_ris * num_elements;
        int theta_end = theta_start + theta_len;

        auto theta_kmn_flat = action.slice(1, theta_start, theta_end).clone();

        int half_len = theta_len / 2;

        theta_kmn_real = theta_kmn_flat.slice(1, 0, half_len)
                         .reshape({batch_size, num_users, num_ris, num_elements});
        theta_kmn_imag = theta_kmn_flat.slice(1, half_len, theta_len)
                         .reshape({batch_size, num_users, num_ris, num_elements});

        int u_start = theta_end;
        int u_end = action.size(1);
        auto u_km_flat = action.slice(1, u_start, u_end).clone();
        u_km = u_km_flat.reshape({batch_size, num_users, num_ris});

        rho_k = rho_k_;
        tau_sched = tau_sched_;

        snr_km = calculate_snr();
        torch::Tensor reward = calculate_reward();
        auto done = reward > 100.0;

        return {get_state(), reward, done};
    }

private:
    torch::Tensor calculate_snr() {
        auto theta_r = theta_kmn_real;
        auto theta_i = theta_kmn_imag;
        auto gr = g_km_real;
        auto gi = g_km_imag;
        auto hr = h_km_real;
        auto hi = h_km_imag;

        // g*h = (gr + jgi)(hr + jhi)
        auto gh_r = gr * hr - gi * hi;
        auto gh_i = gr * hi + gi * hr;

        // theta_ * (g*h)
        auto real = theta_r * gh_r - theta_i * gh_i;
        auto imag = theta_r * gh_i + theta_i * gh_r;

        auto power = real.pow(2) + imag.pow(2);  // |z|^2
        auto channel_gain = power.sum(-1);       // [B, K, M]
        auto rho_expanded = rho_k.unsqueeze(2);  // [B, K, 1]

        return (rho_expanded * channel_gain) / 1e-3;
    }

    torch::Tensor calculate_reward() {
        auto tau_expanded = tau_sched.unsqueeze(2);  // [B, K, 1]
        auto rate = u_km * tau_expanded * bandwidth * torch::log2(1 + snr_km);
        auto sum_rate = rate.sum({1, 2});  // [B]
        return sum_rate;
    }
};

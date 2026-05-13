#pragma once
#include <torch/torch.h>
#include <vector>
#include <cmath>
#include <stdexcept>

//  VectorizedEnvironment

//     Configurable vec_size (batch dimension) set at construction.

//     reset() / step() always work on [vec_size, ...] tensors,
//     so the agent performs ONE forward pass for all sub-envs at
//     once instead of N sequential single-env passes.

//     reset() accepts an optional override so you can reset a
//     single vec_size-1 environment without rebuilding the object.

//     step() returns done as a [B] bool tensor; the caller checks
//     per-environment termination and resets selectively.

class Environment {
public:
    const int    num_users;
    const int    num_ris;
    const int    num_elements;
    const double bandwidth;
    const int    vec_size;   // vectorised batch size

    int state_dim;
    int action_dim;

    // Channel matrices  [B, K, M, N]
    torch::Tensor g_km_real, g_km_imag;
    torch::Tensor h_km_real, h_km_imag;

    // State variables
    torch::Tensor theta_kmn_real, theta_kmn_imag; // [B, K, M, N]
    torch::Tensor snr_km;      // [B, K, M]
    torch::Tensor u_km;        // [B, K, M]
    torch::Tensor tau_sched;   // [B, K]
    torch::Tensor rho_k;       // [B, K]

    //  Constructor
    //  vec_size=1: identical to the original single-env mode.
    //  vec_size>1: vectorised; all tensors gain a batch dim.

    Environment(int num_users_, int num_ris_, int num_elements_,
                double bandwidth_, int vec_size_ = 1)
        : num_users(num_users_)
        , num_ris(num_ris_)
        , num_elements(num_elements_)
        , bandwidth(bandwidth_)
        , vec_size(vec_size_)
    {
        if (vec_size < 1)
            throw std::invalid_argument("vec_size must be >= 1");

        state_dim =
            num_users * num_ris   // snr_km  (flattened)
          + num_users * num_ris   // u_km
          + num_users             // tau_sched
          + num_users;            // rho_k

        action_dim =
            2 * num_users                            // rho_k + tau_sched
          + 2 * num_users * num_ris * num_elements   // theta_kmn (real + imag)
          + num_users * num_ris;                     // u_km
    }

    //  reset:  returns state tensor  [B, state_dim]

    //  Randomises channel matrices and initial state for all B sub-envs in one batched call
    torch::Tensor reset(int override_batch = -1) {
        int B = (override_batch > 0) ? override_batch : vec_size;

        // Channel matrices – fixed per episode, randomised at reset
        g_km_real = torch::rand({B, num_users, num_ris, num_elements});
        g_km_imag = torch::rand({B, num_users, num_ris, num_elements});
        h_km_real = torch::rand({B, num_users, num_ris, num_elements});
        h_km_imag = torch::rand({B, num_users, num_ris, num_elements});

        // State variables
        snr_km       = torch::zeros({B, num_users, num_ris});
        u_km         = torch::zeros({B, num_users, num_ris});
        tau_sched    = torch::rand ({B, num_users});
        rho_k        = torch::rand ({B, num_users});
        theta_kmn_real = torch::rand({B, num_users, num_ris, num_elements});
        theta_kmn_imag = torch::rand({B, num_users, num_ris, num_elements});

        // Assign each user a random initial RIS via scatter
        // u_km[b, k, rand_ris] = 1  for all b, k
        // scatter_() requires a Long index tensor
        auto rand_ris = torch::randint(0, num_ris, {B, num_users},
                                       torch::dtype(torch::kLong));     // [B, K] int64
        u_km.scatter_(/*dim=*/2,
                      rand_ris.unsqueeze(2),                            // [B, K, 1] int64
                      torch::ones({B, num_users, 1}));

        return get_state();
    }


    //  get_state: [B, state_dim]  (flat concatenation)
  
    torch::Tensor get_state() const {
        return torch::cat({
            snr_km.flatten(1),   // [B, K*M]
            u_km.flatten(1),     // [B, K*M]
            tau_sched,           // [B, K]
            rho_k                // [B, K]
        }, /*dim=*/1);
    }

    //  step: {next_state [B,S], reward [B], done [B]}

    //  action shape: [B, action_dim]  (sigmoid-scaled [0,1])

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    step(torch::Tensor action)
    {
        const int B    = action.size(0);
        const int KMN  = num_users * num_ris * num_elements;

        // ---- Parse action tensor ----
        int off = 0;

        rho_k     = action.narrow(1, off, num_users).clone();          off += num_users;
        tau_sched = action.narrow(1, off, num_users).clone();          off += num_users;

        theta_kmn_real = action.narrow(1, off, KMN)
                              .reshape({B, num_users, num_ris, num_elements}).clone();
        off += KMN;
        theta_kmn_imag = action.narrow(1, off, KMN)
                              .reshape({B, num_users, num_ris, num_elements}).clone();
        off += KMN;

        u_km = action.narrow(1, off, num_users * num_ris)
                     .reshape({B, num_users, num_ris}).clone();

        //  Compute next state
        snr_km = calculate_snr();

        torch::Tensor reward = calculate_reward();          // [B]
        // episodes run for a fixed number of steps (controlled by max_steps in the collector)
        // done is always false so the agent never resets mid-episode due to reward magnitude
        torch::Tensor done = torch::zeros({B}, torch::kBool);

        return {get_state(), reward, done};
    }

private:
    //  SNR  =  rho_k * |theta o (g * h)|^2  /  noise_floor
    torch::Tensor calculate_snr() {
        // Complex product: (gr + j gi)(hr + j hi)
        auto gh_r = g_km_real * h_km_real - g_km_imag * h_km_imag;
        auto gh_i = g_km_real * h_km_imag + g_km_imag * h_km_real;

        // theta * (g*h)
        auto prod_r = theta_kmn_real * gh_r - theta_kmn_imag * gh_i;
        auto prod_i = theta_kmn_real * gh_i + theta_kmn_imag * gh_r;

        auto power       = prod_r.pow(2) + prod_i.pow(2);  // [B, K, M, N]
        auto channel_gain = power.sum(-1);                  // [B, K, M]
        auto rho_exp     = rho_k.unsqueeze(2);              // [B, K, 1]

        return (rho_exp * channel_gain) / 1e-3;             // [B, K, M]
    }

    //  Reward  =  sum_{k,m}  u_km * tau_k * B * log2(1 + SNR)
    torch::Tensor calculate_reward() {
        auto tau_exp = tau_sched.unsqueeze(2);              // [B, K, 1]
        auto rate    = u_km * tau_exp * bandwidth * torch::log2(1.0f + snr_km);
        return rate.sum({1, 2}) / bandwidth;                            // [B]
    }
};

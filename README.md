# UL-RIS-DDPG

A C++ reinforcement learning framework that applies **Deep Deterministic Policy Gradient (DDPG)** to jointly optimize uplink transmission in **RIS-assisted MEC networks**. The agent simultaneously tunes user transmission powers, RIS phase shifts, user–RIS association, and uplink scheduling to maximise offloading sum-rate.

---

## Table of Contents

- [Background](#background)
- [Problem Formulation](#problem-formulation)
- [Architecture](#architecture)
- [Multi-threading Design](#multi-threading-design)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Results](#results)
- [Next Steps](#next-steps)

---

## Background

### Reconfigurable Intelligent Surfaces (RIS)

RIS technology uses passive tunable metamaterial arrays to reshape the wireless propagation environment. By controlling the phase shift of each reflecting element, a RIS can constructively combine signals at intended receivers and suppress interference — without generating its own RF emissions. RIS is considered a key enabler for 6G and beyond communication systems, offering coverage extension and energy efficiency gains without the cost of active base station deployment.

### Deep Deterministic Policy Gradient (DDPG)

DDPG is an off-policy actor–critic algorithm for continuous action spaces. Unlike classical optimisation methods (convex relaxation, alternating optimisation) that typically handle RIS phase shifts, power control, and scheduling separately and iteratively, DDPG learns a unified deterministic policy directly from observed network behaviour. This end-to-end approach captures cross-parameter dependencies that decomposed solvers miss.

---

## Problem Formulation

The agent observes the network state and outputs a joint action vector that covers all optimisation variables in a single step.

**State space** `(dim = 4KM)`

| Component | Shape | Description |
|---|---|---|
| `snr_km` | K × M | Per-user per-RIS SNR |
| `u_km` | K × M | User–RIS association matrix |
| `τ_k` | K | Uplink scheduling fractions |
| `ρ_k` | K | Transmit power levels |

**Action space** `(dim = 2K + 2KMN + KM)`

| Component | Shape | Description |
|---|---|---|
| `ρ_k` | K | Transmit power (normalised [0,1]) |
| `τ_k` | K | Scheduling fraction (normalised [0,1]) |
| `Θ_kmn` (real + imag) | 2 × K × M × N | RIS phase shift matrix |
| `u_km` | K × M | User–RIS association |

**Reward**: uplink offloading sum-rate

```
R = Σ_{k,m}  u_km · τ_k · B · log₂(1 + SNR_km)
```

where `SNR_km = ρ_k · |Θ_km ∘ (g_km ⊙ h_km)|² / σ²`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      DDPG Agent                         │
│                                                         │
│   Actor  ──────────────────────────►  Actor Target      │
│   (policy π)      soft update τ       (stable target)   │
│                                                         │
│   Critic ──────────────────────────►  Critic Target     │
│   (Q-value)       soft update τ       (stable target)   │
└────────────────────┬────────────────────────────────────┘
                     │ sample batch
         ┌───────────▼───────────┐
         │     Replay Buffer     │
         │  circular · 20 000    │
         │  non-destructive      │
         │  random sampling      │
         └───────────▲───────────┘
                     │ add experience
       ┌─────────────┼─────────────┐
  Env₁ (×4)    Env₂ (×4)    Env₃ (×4)    ← vectorised environments
  Thread 0     Thread 1     Thread 2      ← collector threads
```

### Networks

Both actor and critic use a 3-layer fully-connected architecture with ReLU activations and a fixed hidden size of 256.

| Network | Input | Output | Activation |
|---|---|---|---|
| Actor | state (30) | action (660) | Sigmoid (bounded [0,1]) |
| Critic | state + action (690) | Q-value (1) | Linear |

Exploration uses **Ornstein–Uhlenbeck noise** added to actor output during collection.

---

## Multi-threading Design

The framework uses a **producer-consumer pipeline** where data collection and network training run concurrently on separate threads.

```
Collector 0 ─┐
Collector 1 ─┼──► ReplayBuffer ──► Trainer
Collector 2 ─┘   (shared, thread-safe)
```

### Thread Safety

**ReplayBuffer**
- Pre-allocated circular `std::vector` — no heap allocation after init
- Tensor `.clone()` done *before* acquiring any lock
- Write index protected by a short-held `std::mutex`; slot writes use 16 striped mutexes so concurrent writers to different slots do not block each other
- `sample()` is non-destructive (random index draw) so collection and training never compete for the same entries

**Actor Network**
- Guarded by `std::shared_mutex`
- `select_action()` acquires a `shared_lock` — any number of collector threads can run inference simultaneously
- The actor optimiser step and the subsequent `soft_update` of the actor target both run inside a single `unique_lock`, preventing collectors from reading a half-updated weight tensor

**Critic Network**
- Accessed exclusively by the training thread — no locking required

**Shutdown**
- A single `std::atomic<bool> stop_flag` is set by the trainer when training completes; collectors check it at the top of each episode loop and exit cleanly

---

## Project Structure

```
.
├── main.cpp      # Entry point, thread launch, hyperparameters
├── DDPG.cpp      # ReplayBuffer, Actor, Critic, DDPG agent, OUNoise
├── env.cpp       # Vectorised RIS-MEC environment, SNR & reward computation
└── README.md
```

---

## Requirements

| Dependency | Version |
|---|---|
| C++ standard | C++17 or later |
| LibTorch | ≥ 1.10 |
| CMake | ≥ 3.14 |

---

## Configuration

All hyperparameters are `constexpr` constants in `main.cpp`.

| Parameter | Default | Description |
|---|---|---|
| `num_users` | 5 | Number of UL users (K) |
| `num_ris` | 2 | Number of RIS surfaces (M) |
| `num_elements` | 32 | Elements per RIS (N) |
| `bandwidth` | 1e6 Hz | System bandwidth (B) |
| `num_collectors` | 3 | Parallel collector threads |
| `vec_size` | 4 | Vectorised sub-environments per thread |
| `max_steps` | 50 | Steps per episode |
| `train_iters` | 2000 | Total training gradient steps |
| `batch_size` | 64 | Mini-batch size |
| `buf_capacity` | 20 000 | Replay buffer capacity |
| `actor_lr` | 5e-5 | Actor learning rate |
| `critic_lr` | 1e-4 | Critic learning rate |
| `tau` | 1e-3 | Soft update coefficient |
| `gamma` | 0.99 | Discount factor |

---

## Results

In the following examlpe Training converges with a consistent upward trend in both episode reward and Q-value estimates. Example run (`K=5, M=2, N=32`):

```
Step    0 | Episodes:    3 | Avg Ep Reward:   497 | A-Loss:   -26 | C-Loss:  28580 | Avg Q:   -65
Step  500 | Episodes:  624 | Avg Ep Reward:  5174 | A-Loss: -2047 | C-Loss:   5680 | Avg Q:  2050
Step 1000 | Episodes: 1308 | Avg Ep Reward:  6330 | A-Loss: -2435 | C-Loss:   6722 | Avg Q:  2403
Step 1500 | Episodes: 2107 | Avg Ep Reward:  6730 | A-Loss: -2912 | C-Loss:  15438 | Avg Q:  2932
Step 1950 | Episodes: 2799 | Avg Ep Reward:  6893 | A-Loss: -3259 | C-Loss:   7337 | Avg Q:  3251
```


---

## Next Steps

Since the training step is the throughput bottleneck, profiling shows collectors generate data faster than the trainer consumes it, meaning additional collector threads add contention on `actor_rw_` without meaningful gain, the most impactful parallelism improvements target the update loop itself. Two directions are viable on CPU. First, **data-parallel gradient computation**: split each mini-batch into N equal sub-batches, compute forward and backward passes concurrently on separate threads, then reduce the resulting gradients before calling `optimizer.step()` This scales linearly with available cores up to the point where synchronisation overhead dominates. Second, **pipelined actor/critic updates**: because the critic update does not require the actor lock, critic and actor gradient steps can be overlapped across consecutive training iterations on two dedicated threads, with careful read/write ordering on the shared critic weights, this halves the effective latency of each update at the cost of introducing a one-step lag between the critic seen by the actor and the critic seen by the Bellman target.

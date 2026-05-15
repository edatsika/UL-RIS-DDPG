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
- [Changes log](#changes-log)

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
| `batch_size` | 256 | Mini-batch size |
| `buf_capacity` | 50 000 | Replay buffer capacity |
| `actor_lr` | 5e-5 | Actor learning rate |
| `critic_lr` | 1e-4 | Critic learning rate |
| `tau` | 5e-4 | Soft update coefficient |
| `gamma` | 0.99 | Discount factor |

---

## Results

Training converges with a consistent upward trend in both episode reward and Q-value estimates. Example run (`K=5, M=2, N=32`):

```

| Step | Episodes | Avg Ep Reward | A-Loss | C-Loss | Avg Q |
|-----:|--------:|-------------:|-------:|-------:|------:|
|    0 |       0 |            0 |    153 |    529 |  -194 |
|  500 |      96 |         3623 |   -622 |     78 |   619 |
| 1000 |     224 |         4483 |   -935 |    119 |   911 |
| 1500 |     346 |         5376 |  -1776 |    150 |  1815 |
| 1950 |     448 |         5757 |  -2589 |     96 |  2562 |
| 5000 |    1159 |         6593 |  -4242 |     53 |  4232 |
| 9950 |    2461 |         6883 |  -6754 |     59 |  6714 |
```

---

## Changes log

### Thread Safety

- Data race on Actor/Critic weights between `select_action()` and `update()`: `std::shared_mutex`: `shared_lock` for collectors, `unique_lock` for trainer 
- Concurrent `buffer.add()` from multiple collectors: `write_mtx_` for metadata + `stripe_mtx_[16]` for individual slots 
- Concurrent `buffer.sample()` during `add()`: Per-slot `stripe_mtx_` + dedicated `rng_mtx_` 
- Deadlock on training end: `sample()` blocked in `cv_.wait()` indefinitely: `notify_stop()` wakes all waiters; `stop_flag` checked in `cv_.wait` 
- Shared `OUNoise` across all collector threads and serialised sampling, correlated exploration: one `OUNoise` instance per sub-environment, owned by its collector thread, mutex removed 
- Identical noise added to all `vec_size` environments per step via `expand_as()`: per-environment noise sampled independently, stacked into `[B, action_dim]` tensor

<!-- ### Training Stability

- Critic loss: `mse_loss` → `smooth_l1_loss` (Huber): MSE amplifies large TD errors quadratically, causing critic loss spikes up to 515k; Huber clips gradient magnitude for large errors
- `batch_size`: 64 → 256: more stable gradient estimates for 660-dimensional action space; reduces variance per update step 
- `buf_capacity`: 20k → 50k: prevents recent experiences from overwriting older ones too quickly; improves sample diversity
- `actor_lr`: 5e-5 → 5e-6  Slows actor relative to critic, reducing Q-value divergence 
- `critic_lr`: 1e-4 → 1e-4 (unchanged): Huber loss already stabilises critic; aggressive lr no longer needed to chase spikes
- `tau`: 1e-3 → 5e-4: Slower target network updates reduce bootstrap instability-->

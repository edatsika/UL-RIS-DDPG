# UL-RIS-DDPG

UL‑RIS‑DDPG is a C++ based algorithmic framework that applies Deep Deterministic Policy Gradient (DDPG) reinforcement learning to jointly optimize key parameters in uplink (UL) MEC networks assisted by Reconfigurable Intelligent Surfaces (RIS).
It joinly tunes users’ transmission powers, design of RIS phase shifts, user‑RIS association, and uplink transmission scheduling to significantly improve the network’s offloading sum‑rate performance.

---

## 🧩 Background

**Reconfigurable Intelligent Surfaces (RIS):**

RIS technology uses tunable metamaterial arrays to modify wireless propagation environments, enhancing signal quality and coverage. They are key enablers for 6G and beyond communication systems.

**Deep Deterministic Policy Gradient (DDPG):**

DDPG is a deep reinforcement learning algorithm designed for continuous action spaces. It uses an actor–critic architecture to learn deterministic control policies. Unlike classical optimization techniques that separately handle RIS phase shifts, transmit power control and other parameters, this approach learns policies directly from network behavior and tunes the related parameters accordingly.

---
## ⚡ Multi-threading Support

The project includes thread-safe components to enable multi-threaded training:

- **Replay Buffer**: Uses mutexes and condition variables to safely allow multiple threads to add and sample experiences concurrently.
- **Actor Network Access**: Action selection is protected by a mutex, ensuring safe concurrent queries in multi-threaded environments.

This design allows parallel interaction with multiple environment instances, improving training efficiency.

---

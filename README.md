# Autonomous Racetrack Agents 🏎️

This project implements and compares various **Reinforcement Learning (RL)** and **Imitation Learning** agents on the `racetrack-v0` environment from **Highway-Env**. The agents range from classic tabular methods to deep reinforcement learning approaches.

---

## 🚀 Usage

### 1. Create the Demo Runner

To run all agents sequentially, create a file named **`demo_all.py`** with the following content:

### 2. Run the Demo

Execute the runner from your terminal:

```bash
python demo_all.py
```

---

## 🧠 Agent Architectures

### 1. Actor-Critic (PPO)

**File:** `actor_critique.py`

A deep reinforcement learning agent using **Proximal Policy Optimization (PPO)** for continuous control.

* **Action Space:**

  * Acceleration ∈ [-4, 4]
  * Steering ∈ [-0.78, 0.78]
* **Reward Function:**

  * Speed: 0.6
  * Lane centering: 0.3
  * Steering smoothness penalty: 0.1
* **Training:**

  * "Good" model trained for ~60,000 timesteps

---

### 2. Behavior Cloning

**File:** `3_demo_final.py`

An imitation learning agent that mimics expert driving behavior.

* **Network:** Feed-forward neural network (Linear → ReLU → Linear)
* **Input:** Flattened kinematic observation vector
* **Execution:**

  * Loads pretrained weights from `model_iteratia_3.pth`
  * Runs deterministically with no further learning

---

### 3. Dyna-Q (Model-Based)

**File:** `last_version_1.py`

A tabular RL method combining real experience with simulated planning.

* **Planning:**

  * 50 simulated (imagined) steps per real environment step
* **State Representation:**

  * Continuous state (x, y, vx, vy, h) discretized (e.g., x ÷ 4)
* **Actions:**

  * Left, Idle, Right, Gas, Brake

---

### 4. Deep Q-Network (DQN)

**File:** `run_dqn.py`

A value-based deep RL agent operating on spatial observations.

* **Observation:**

  * Occupancy Grid (2D spatial representation of surroundings)
* **Configuration:**

  * Action bins: 5 (bucketed continuous actions)
  * Replay buffer size: 4,000
  * Epsilon decay: 0.995

---

### 5. Q-Learning (Tabular)

**File:** `qlearning/run.py`

A classic tabular Q-learning agent.

* **Discretization:**

  * Observation buckets: [8, 4, 5]
* **Policy:**

  * Epsilon = 0.0 during demo (purely greedy)
* **Model:**

  * Loads Q-table from `final_q_table.pkl`

---

## 🛠️ Requirements

* Python 3.8+
* gymnasium
* highway-env
* torch
* stable-baselines3
* matplotlib
* numpy

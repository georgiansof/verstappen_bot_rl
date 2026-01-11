"""
Evaluate and Compare Trained RL Models
======================================
This script loads pre-trained Q-tables and evaluates each model's performance
by running multiple test episodes. No training is performed.
"""

import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random
from collections import defaultdict

# === CONFIGURATION ===
MODELS = {
    "Basic V1": "model_basic_v1.pkl",
    "Basic V2": "model_basic_v2.pkl",
    "Dyna-Q+": "model_dyna_q_plus.pkl",
    "Prioritized Sweeping": "model_prioritized_sweeping.pkl",
    "Traffic Aware": "model_traffic_aware.pkl",
}

TEST_EPISODES = 100  # Number of episodes to evaluate each model
SMOOTHING_WINDOW = 10  # For plotting


# === ENVIRONMENT ===
class DiscretizedRacetrack:
    def __init__(self, render_mode=None):
        config = {
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy", "heading"],
                "normalize": False
            },
            "duration": 40,
            "simulation_frequency": 15,
            "policy_frequency": 5,
        }
        self.env = gym.make("racetrack-v0", render_mode=render_mode, config=config)
        self.action_space = 5

    def reset(self):
        obs, _ = self.env.reset()
        return self._discretize(obs)

    def step(self, action_idx):
        if action_idx == 0:   cont_action = [0.0, 0.5]
        elif action_idx == 1: cont_action = [0.0, 0.0]
        elif action_idx == 2: cont_action = [0.0, -0.5]
        elif action_idx == 3: cont_action = [0.7, 0.0]
        elif action_idx == 4: cont_action = [-0.7, 0.0]
        else: cont_action = [0.0, 0.0]

        obs, reward, done, truncated, _ = self.env.step(cont_action)
        
        if done:
            reward = -20
        else:
            reward += 0.1
        
        return self._discretize(obs), reward, done or truncated

    def _discretize(self, obs):
        ego = obs[0]
        x, y, vx, vy, h = ego[0], ego[1], ego[2], ego[3], ego[4]
        return (int(x/4), int(y/4), int(vx/1), int(vy/1), int(h/0.5))

    def close(self):
        self.env.close()


# === AGENT ===
class EvaluationAgent:
    def __init__(self, actions):
        self.q_table = defaultdict(float)
        self.actions = range(actions)

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.q_table = defaultdict(float, pickle.load(f))
        return len(self.q_table)

    def choose_action(self, state):
        q_vals = [self.q_table[(state, a)] for a in self.actions]
        max_q = max(q_vals)
        ties = [a for a, q in zip(self.actions, q_vals) if q == max_q]
        return random.choice(ties)


# === UTILITIES ===
def moving_average(data, window=50):
    """Smooth data using a moving average."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def evaluate_model(model_path, num_episodes):
    """Evaluate a single model and return episode rewards."""
    env = DiscretizedRacetrack(render_mode=None)
    agent = EvaluationAgent(env.action_space)
    
    num_states = agent.load_model(model_path)
    rewards = []
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            state, reward, done = env.step(action)
            total_reward += reward
        
        rewards.append(total_reward)
        
        if (ep + 1) % 25 == 0:
            print(f"      Episode {ep+1}/{num_episodes} - Reward: {total_reward:.1f}")
    
    env.close()
    return rewards, num_states


# === MAIN ===
def main():
    print("=" * 60)
    print("  MODEL EVALUATION AND COMPARISON")
    print("=" * 60)
    
    results = {}
    state_counts = {}
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    # Evaluate each model
    for i, (name, filename) in enumerate(MODELS.items()):
        if not os.path.exists(filename):
            print(f"\n⚠️  {name}: File '{filename}' not found, skipping...")
            continue
        
        print(f"\n📊 Evaluating: {name}")
        print(f"   Loading: {filename}")
        
        rewards, num_states = evaluate_model(filename, TEST_EPISODES)
        results[name] = rewards
        state_counts[name] = num_states
        
        print(f"   ✓ Complete! States learned: {num_states}")
    
    if not results:
        print("\n❌ No models found to evaluate!")
        return
    
    # === SUMMARY TABLE ===
    print("\n" + "=" * 60)
    print("  PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'States':>8} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 60)
    
    for name, rewards in results.items():
        max_r = max(rewards)
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        states = state_counts[name]
        print(f"{name:<25} {states:>8} {max_r:>10.1f} {mean_r:>10.1f} {std_r:>10.1f}")
    
    print("-" * 60)
    
    # Find best model
    best_model = max(results.keys(), key=lambda k: np.mean(results[k]))
    print(f"\n🏆 Best Average Performance: {best_model} ({np.mean(results[best_model]):.1f})")
    
    # === PLOTTING ===
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Episode rewards (smoothed)
    ax1 = axes[0]
    for i, (name, rewards) in enumerate(results.items()):
        color = colors[i % len(colors)]
        smoothed = moving_average(rewards, SMOOTHING_WINDOW)
        ax1.plot(smoothed, label=name, color=color, linewidth=2)
    
    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Total Reward (Smoothed)", fontsize=12)
    ax1.set_title("Evaluation Performance Over Episodes", fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison
    ax2 = axes[1]
    box_data = [results[name] for name in results.keys()]
    box_labels = list(results.keys())
    
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)
    
    ax2.set_ylabel("Total Reward", fontsize=12)
    ax2.set_title("Reward Distribution by Model", fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Comparison of Model-Based RL Variants on Racetrack", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n📈 Graph saved to: model_comparison.png")
    
    plt.show()


if __name__ == "__main__":
    main()

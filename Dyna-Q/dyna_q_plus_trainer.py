import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import os
from collections import defaultdict

# === 1. MEDIUL DE JOC (Discretizat) ===
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
        if action_idx == 0:   cont_action = [0.0, 0.5]   # Stanga
        elif action_idx == 1: cont_action = [0.0, 0.0]   # Idle
        elif action_idx == 2: cont_action = [0.0, -0.5]  # Dreapta
        elif action_idx == 3: cont_action = [0.7, 0.0]   # Gaz
        elif action_idx == 4: cont_action = [-0.7, 0.0]  # Frana
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

# === 2. AGENTUL DYNA-Q+ (Cu bonus de explorare) ===
class DynaQPlusAgent:
    """
    Dyna-Q+ adaugă un bonus pentru stările nevizitate de mult timp.
    Asta încurajează agentul să exploreze zone "uitate" ale mediului.
    """
    def __init__(self, actions, epsilon=0.15, alpha=0.1, gamma=0.9, planning_steps=50, kappa=0.001):
        self.q_table = defaultdict(float)
        self.model = {}
        self.actions = range(actions)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.planning_steps = planning_steps
        
        # === NOU: Pentru Dyna-Q+ ===
        self.kappa = kappa  # Factor bonus pentru explorare
        self.time_since_visit = defaultdict(int)  # Timp de când n-am vizitat (s, a)
        self.time_step = 0  # Contor global de timp

    def choose_action(self, state, force_greedy=False):
        if not force_greedy and random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        
        q_vals = [self.q_table[(state, a)] for a in self.actions]
        max_q = max(q_vals)
        ties = [a for a, q in zip(self.actions, q_vals) if q == max_q]
        return random.choice(ties)

    def update(self, state, action, reward, next_state):
        self.time_step += 1
        
        # Q-Learning standard
        max_next_q = max([self.q_table[(next_state, a)] for a in self.actions])
        current_q = self.q_table[(state, action)]
        self.q_table[(state, action)] += self.alpha * (reward + self.gamma * max_next_q - current_q)

        # Actualizăm modelul
        self.model[(state, action)] = (reward, next_state)
        
        # Resetăm timpul pentru această pereche (s, a)
        self.time_since_visit[(state, action)] = 0
        
        # Incrementăm timpul pentru toate celelalte perechi cunoscute
        for key in self.model.keys():
            if key != (state, action):
                self.time_since_visit[key] += 1

        # === PLANNING CU BONUS (Dyna-Q+) ===
        for _ in range(self.planning_steps):
            if not self.model:
                break
            s, a = random.choice(list(self.model.keys()))
            r, next_s = self.model[(s, a)]
            
            # Bonus pentru explorare: cu cât nu am vizitat mai demult, cu atât mai mare
            exploration_bonus = self.kappa * np.sqrt(self.time_since_visit[(s, a)])
            
            max_next = max([self.q_table[(next_s, act)] for act in self.actions])
            # Folosim reward + bonus în update
            self.q_table[(s, a)] += self.alpha * (
                r + exploration_bonus + self.gamma * max_next - self.q_table[(s, a)]
            )

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.q_table = defaultdict(float, pickle.load(f))

# === 3. ANTRENAMENT ===
def antreneaza_si_salveaza():
    PLANNING = 40
    EPISODES = 10000
    KAPPA = 0.001  # Factor de explorare pentru Dyna-Q+
    
    print(f">>> Încep antrenarea Dyna-Q+ ({EPISODES} episoade, kappa={KAPPA})...")
    env = DiscretizedRacetrack(render_mode=None)
    agent = DynaQPlusAgent(env.action_space, planning_steps=PLANNING, kappa=KAPPA)
    
    rewards = []
    best_avg_reward = -9999
    
    for ep in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        rewards.append(total_reward)

        avg_score = np.mean(rewards[-20:]) if len(rewards) > 20 else -999
        if avg_score > best_avg_reward and ep > 50:
            best_avg_reward = avg_score
            agent.save_model("model_dyna_q_plus.pkl")
        
        if (ep+1) % 100 == 0:
            print(f"   Episod {ep+1}/{EPISODES} | Avg(20): {avg_score:.2f} | eps={agent.epsilon:.2f}")

    env.close()
    
    plt.plot(np.convolve(rewards, np.ones(20)/20, mode='valid'))
    plt.title("Performanța Dyna-Q+ (Cu Bonus Explorare)")
    plt.ylabel("Reward")
    plt.xlabel("Episoade")
    plt.grid()
    plt.savefig("rezultat_dyna_q_plus.png")
    print("\n>>> Antrenament Dyna-Q+ gata! Grafic salvat.")

# === 4. DEMO VIZUAL ===
def ruleaza_demo():
    print("\n>>> DEMO VIZUAL DYNA-Q+...")
    if not os.path.exists("model_dyna_q_plus.pkl"):
        print("Nu am găsit modelul salvat!")
        return

    env = DiscretizedRacetrack(render_mode="human")
    agent = DynaQPlusAgent(env.action_space)
    agent.load_model("model_dyna_q_plus.pkl")
    
    for i in range(5):
        state = env.reset()
        done = False
        total = 0
        print(f"   -> Cursa {i+1} start...")
        while not done:
            action = agent.choose_action(state, force_greedy=True)
            state, reward, done = env.step(action)
            env.env.render()
            total += reward
        print(f"   -> Cursa {i+1} gata. Scor: {total:.1f}")
    
    env.close()

if __name__ == "__main__":
    antreneaza_si_salveaza()
    ruleaza_demo()

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
        # Mapare acțiuni discrete -> continue
        if action_idx == 0:   cont_action = [0.0, 0.5]  # Stanga
        elif action_idx == 1: cont_action = [0.0, 0.0]  # Idle
        elif action_idx == 2: cont_action = [0.0, -0.5] # Dreapta
        elif action_idx == 3: cont_action = [0.7, 0.0]  # Gaz
        elif action_idx == 4: cont_action = [-0.7, 0.0] # Frana
        else: cont_action = [0.0, 0.0]

        obs, reward, done, truncated, _ = self.env.step(cont_action)
        
        # Reward Shaping: Îl încurajăm să meargă, nu să stea
        if done:
            reward = -20 # Penalizare accident
        else:
            # Mic bonus pentru viteză ca să nu stea pe loc
            reward += 0.1 
        
        return self._discretize(obs), reward, done or truncated

    def _discretize(self, obs):
        ego = obs[0]
        x, y, vx, vy, h = ego[0], ego[1], ego[2], ego[3], ego[4]
        # Tuning fin pentru discretizare
        return (int(x/4), int(y/4), int(vx/1), int(vy/1), int(h/0.5))

    def close(self):
        self.env.close()

# === 2. AGENTUL DYNA-Q ===
class DynaQAgent:
    def __init__(self, actions, epsilon=0.15, alpha=0.1, gamma=0.9, planning_steps=50):
        self.q_table = defaultdict(float)
        self.model = {} 
        self.actions = range(actions)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.planning_steps = planning_steps

    def choose_action(self, state, force_greedy=False):
        # Dacă testăm (force_greedy), nu mai explorăm random
        if not force_greedy and random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        
        q_vals = [self.q_table[(state, a)] for a in self.actions]
        max_q = max(q_vals)
        ties = [a for a, q in zip(self.actions, q_vals) if q == max_q]
        return random.choice(ties)

    def update(self, state, action, reward, next_state):
        # Q-Learning
        max_next_q = max([self.q_table[(next_state, a)] for a in self.actions])
        current_q = self.q_table[(state, action)]
        self.q_table[(state, action)] += self.alpha * (reward + self.gamma * max_next_q - current_q)

        # Model Learning
        self.model[(state, action)] = (reward, next_state)

        # Planning
        for _ in range(self.planning_steps):
            if not self.model: break
            s, a = random.choice(list(self.model.keys()))
            r, next_s = self.model[(s, a)]
            max_next = max([self.q_table[(next_s, act)] for act in self.actions])
            self.q_table[(s, a)] += self.alpha * (r + self.gamma * max_next - self.q_table[(s, a)])

    def save_model(self, filename):
        # Salvăm tabelul Q într-un fișier
        with open(filename, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.q_table = defaultdict(float, pickle.load(f))

# === 3. ANTRENAMENT ===
def antreneaza_si_salveaza():
    PLANNING = 40
    EPISODES = 10000 # Mărit pentru performanță
    
    print(f">>> Încep antrenarea Dyna-Q ({EPISODES} episoade)... Nu închide!")
    env = DiscretizedRacetrack(render_mode=None) # Fără grafică pt viteză
    agent = DynaQAgent(env.action_space, planning_steps=PLANNING)
    
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
        
        # Scade epsilon în timp (explorăm mai puțin pe măsură ce învățăm)
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        rewards.append(total_reward)

        # Salvăm cel mai bun model de până acum
        avg_score = np.mean(rewards[-20:]) if len(rewards) > 20 else -999
        if avg_score > best_avg_reward and ep > 50:
            best_avg_reward = avg_score
            agent.save_model("best_dyna_q.pkl")
        
        if (ep+1) % 100 == 0:
            print(f"   Episod {ep+1}/{EPISODES} | Reward Mediu (ultimele 20): {avg_score:.2f} | Epsilon: {agent.epsilon:.2f}")

    env.close()
    
    # Desenăm graficul
    plt.plot(np.convolve(rewards, np.ones(20)/20, mode='valid'))
    plt.title("Performanta Dyna-Q (Antrenare)")
    plt.ylabel("Reward")
    plt.xlabel("Episoade")
    plt.grid()
    plt.savefig("rezultat_dyna_q.png")
    print("\n>>> Antrenament gata! Grafic salvat.")

# === 4. DEMO VIZUAL ===
def ruleaza_demo():
    print("\n>>> PORNIRE DEMO VIZUAL CU CEL MAI BUN MODEL...")
    if not os.path.exists("best_dyna_q.pkl"):
        print("Nu am găsit modelul salvat!")
        return

    # Aici activăm grafica ('human')
    env = DiscretizedRacetrack(render_mode="human")
    agent = DynaQAgent(env.action_space)
    agent.load_model("model_basic_v2.pkl")
    
    for i in range(5): # 5 curse demonstrative
        state = env.reset()
        done = False
        total = 0
        print(f"   -> Cursa {i+1} start...")
        while not done:
            # force_greedy=True înseamnă că folosim doar ce am învățat, nu explorăm
            action = agent.choose_action(state, force_greedy=True)
            state, reward, done = env.step(action)
            env.env.render()
            total += reward
        print(f"   -> Cursa {i+1} gata. Scor: {total:.1f}")
    
    env.close()

if __name__ == "__main__":
    # Pasul 1: Antrenăm "în întuneric" (rapid)
    antreneaza_si_salveaza()
    
    # Pasul 2: Vedem rezultatul cu grafică
    ruleaza_demo()
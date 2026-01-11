import gymnasium as gym
import highway_env
import numpy as np
import pickle
import os
import random
from collections import defaultdict

# === CONFIGURĂRI ===
MODEL_FILE = "model_dyna_q_plus.pkl"  # Numele fișierului tău salvat
RENDER_MODE = "human"           # "human" pentru a vedea jocul

# === 1. CLASA MEDIULUI (Trebuie să fie identică cu cea de antrenare) ===
class DiscretizedRacetrack:
    def __init__(self, render_mode=None):
        config = {
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy", "heading"],
                "normalize": False
            },
            "duration": 60, # Îl lăsăm să meargă mai mult timp în demo
            "simulation_frequency": 15,
            "policy_frequency": 5,
        }
        self.env = gym.make("racetrack-v0", render_mode=render_mode, config=config)
        self.action_space = 5 
    
    def reset(self):
        obs, _ = self.env.reset()
        return self._discretize(obs)

    def step(self, action_idx):
        # Mapare acțiuni (aceeași ca la antrenare)
        if action_idx == 0:   cont_action = [0.0, 0.5]  # Stanga
        elif action_idx == 1: cont_action = [0.0, 0.0]  # Idle
        elif action_idx == 2: cont_action = [0.0, -0.5] # Dreapta
        elif action_idx == 3: cont_action = [0.7, 0.0]  # Gaz
        elif action_idx == 4: cont_action = [-0.7, 0.0] # Frana
        else: cont_action = [0.0, 0.0]

        obs, reward, done, truncated, _ = self.env.step(cont_action)
        return self._discretize(obs), reward, done or truncated

    def _discretize(self, obs):
        ego = obs[0]
        x, y, vx, vy, h = ego[0], ego[1], ego[2], ego[3], ego[4]
        # Aceeași discretizare folosită la antrenare
        return (int(x/4), int(y/4), int(vx/1), int(vy/1), int(h/0.5))

    def close(self):
        self.env.close()

# === 2. CLASA AGENTULUI (Doar partea de încărcare și decizie) ===
class DynaQAgent:
    def __init__(self, actions):
        self.q_table = defaultdict(float)
        self.actions = range(actions)

    def load_model(self, filename):
        print(f"--> Încărcăm modelul din: {filename}")
        with open(filename, "rb") as f:
            self.q_table = defaultdict(float, pickle.load(f))
        print(f"--> Model încărcat! Număr stări învățate: {len(self.q_table)}")

    def choose_action(self, state):
        # Aici e secretul: force_greedy este implicit.
        # Nu mai explorăm (random), alegem doar ce știm că e cel mai bine.
        q_vals = [self.q_table[(state, a)] for a in self.actions]
        max_q = max(q_vals)
        ties = [a for a, q in zip(self.actions, q_vals) if q == max_q]
        return random.choice(ties)

# === 3. BUCLA INFINITĂ ===
def run_endless():
    if not os.path.exists(MODEL_FILE):
        print(f"EROARE: Nu găsesc fișierul '{MODEL_FILE}'. Verifică numele!")
        return

    # Inițializăm mediul și agentul
    env = DiscretizedRacetrack(render_mode=RENDER_MODE)
    agent = DynaQAgent(env.action_space)
    
    # Încărcăm creierul
    try:
        agent.load_model(MODEL_FILE)
    except Exception as e:
        print(f"Eroare la încărcarea modelului: {e}")
        return

    print("\n>>> START SIMULARE INFINITĂ (Apasă Ctrl+C pentru a opri) <<<")
    
    episod_cnt = 1
    try:
        while True: # Loop infinit
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                # Agentul alege cea mai bună acțiune
                action = agent.choose_action(state)
                
                # Executăm
                state, reward, done = env.step(action)
                
                # Afișăm (Render este automat în step dacă render_mode='human', 
                # dar apelăm și explicit pentru siguranță pe unele sisteme)
                env.env.render()
                
                total_reward += reward
                steps += 1
            
            print(f"Cursa {episod_cnt} terminată | Scor: {total_reward:.1f} | Pași: {steps}")
            episod_cnt += 1

    except KeyboardInterrupt:
        print("\n\n>>> Simulare oprită de utilizator. La revedere!")
        env.close()

if __name__ == "__main__":
    run_endless()
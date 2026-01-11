import torch
import torch.nn as nn
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import highway_env
import os

ENV_NAME = "racetrack-v0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CONFIGURARE RACING ---
CONFIG_RACING = {
    "action": {
        "type": "ContinuousAction", "longitudinal": True, "lateral": True,
        "acceleration_range": [-4, 4], "steering_range": [-0.78, 0.78]
    },
    "speed_limit": 30, "lane_centering_cost": 4
}

# --- WRAPPER EVALUARE ---
class RacingRewardWrapper(gym.Wrapper):
    def __init__(self, env): super().__init__(env)
    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        v = self.env.unwrapped.vehicle
        cfg = self.env.unwrapped.config
        
        speed_rw = v.speed / cfg["speed_limit"] if v.speed > 0 else -0.5
        _, lat = v.lane.local_coordinates(v.position)
        center_rw = 1 / (1 + cfg["lane_centering_cost"] * lat**2)
        steer_rw = -(action[1]**2)
        
        total = 0.6*speed_rw + 0.3*center_rw + 0.1*steer_rw
        if v.crashed: total = -1.0
        return obs, total * float(v.on_road), done, truncated, info

class BehaviorCloningNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BehaviorCloningNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x): return self.net(x)

def evalueaza_model(nume_model):
    if not os.path.exists(nume_model): return 0
    
    raw_env = gym.make(ENV_NAME, render_mode=None, config=CONFIG_RACING)
    env = RacingRewardWrapper(raw_env)
    
    obs, _ = env.reset()
    input_dim = np.prod(obs.shape)
    output_dim = env.action_space.shape[0]
    model = BehaviorCloningNet(input_dim, output_dim).to(DEVICE)
    try: model.load_state_dict(torch.load(nume_model, map_location=DEVICE))
    except: return 0

    scores = []
    for _ in range(5):
        obs, _ = env.reset()
        done, truncated, ep_score = False, False, 0
        while not (done or truncated):
            obs_t = torch.FloatTensor(obs).reshape(1, -1).to(DEVICE)
            with torch.no_grad(): action = model(obs_t).cpu().numpy()[0]
            obs, r, done, truncated, _ = env.step(action)
            ep_score += r
        scores.append(ep_score)
    env.close()
    return np.mean(scores)

def plot_all():
    # 1. Bar Chart
    print("Generez Bar Chart...")
    models = ["model_iteratia_1.pth", "model_iteratia_2.pth", "model_iteratia_3.pth"]
    labels = ["Random", "Mediu", "Expert"]
    vals = [evalueaza_model(m) for m in models]
    
    plt.figure(figsize=(10,6))
    bars = plt.bar(labels, vals, color=['red','orange','green'])
    plt.title("Performanța RACING (Viteză + Precizie)")
    plt.ylabel("Scor Mediu")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom', fontweight='bold')
    plt.savefig("grafic_evaluare.png")
    
    # 2. Histograma Steering
    print("Generez Histograma...")
    plt.figure(figsize=(10,6))
    for f, c, l in zip(["dataset_slab.pkl", "dataset_expert.pkl"], ['red', 'green'], ["Random", "Expert"]):
        try:
            with open(f, "rb") as file:
                data = pickle.load(file)
                if data["actions"].shape[1] > 1:
                    # Coloana 1 este Steering
                    plt.hist(data["actions"][:, 1], bins=50, alpha=0.5, color=c, label=l, density=True)
        except: pass
    plt.title("Distribuția Volanului (Steering)")
    plt.legend()
    plt.savefig("grafic_histograma.png")

if __name__ == "__main__":
    plot_all()
    print("Gata! Graficele sunt actualizate.")
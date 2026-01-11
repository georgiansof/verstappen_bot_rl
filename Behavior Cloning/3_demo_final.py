import gymnasium as gym
import highway_env
import torch
import torch.nn as nn
import numpy as np

ENV_NAME = "racetrack-v0"
MODEL_PATH = "model_iteratia_3.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CONFIGURARE RACING ---
CONFIG_RACING = {
    "action": {
        "type": "ContinuousAction", 
        "longitudinal": True, 
        "lateral": True,
        "acceleration_range": [-4, 4], 
        "steering_range": [-0.78, 0.78]
    },
    "duration": 500,               
    "speed_limit": 30, 
    "lane_centering_cost": 4,
    "screen_width": 1000,          
    "screen_height": 1000
}

# --- WRAPPER PENTRU DISPLAY SCOR ---
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

def ruleaza_demo():
    print(f"--- Încărcăm modelul: {MODEL_PATH} ---")
    
    raw_env = gym.make(ENV_NAME, render_mode="human", config=CONFIG_RACING)
    env = RacingRewardWrapper(raw_env)
    
    obs, _ = env.reset()
    input_dim = np.prod(obs.shape)
    output_dim = env.action_space.shape[0] 
    
    model = BehaviorCloningNet(input_dim, output_dim).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except:
        print("Nu găsesc modelul! Ruleaza scriptul 2.")
        return

    print("\n--- START CURSE (Apasă Ctrl+C în terminal pentru a opri) ---")
    
    for ep in range(5):
        obs, _ = env.reset()
        done = False
        truncated = False
        score = 0
        steps = 0
        
        while not (done or truncated):
            obs_t = torch.FloatTensor(obs).reshape(1, -1).to(DEVICE)
            with torch.no_grad():
                action = model(obs_t).cpu().numpy()[0]
            
            obs, r, done, truncated, info = env.step(action)
            env.render()
            score += r
            steps += 1
            
        motiv = "ACCIDENT" if env.unwrapped.vehicle.crashed else "TIMP EXPIRAT (Succes)"
        print(f"Cursa {ep+1}: Scor = {score:.2f} | Pași: {steps} | Rezultat: {motiv}")

    env.close()

if __name__ == "__main__":
    ruleaza_demo()
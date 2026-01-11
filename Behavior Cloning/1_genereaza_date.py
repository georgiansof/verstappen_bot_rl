import gymnasium as gym
import highway_env
import numpy as np
import pickle
import torch
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# === SETĂRI ===
N_CPU = 8  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENV_NAME = "racetrack-v0"

print(f"--> [Hardware] Folosim {N_CPU} procese pe {DEVICE.upper()}")


CONFIG_RACING = {
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": True,           
        "acceleration_range": [-4, 4],
        "steering_range": [-0.78, 0.78]
    },
    "duration": 300,
    "speed_limit": 30,
    "lane_centering_cost": 4
}

# --- WRAPPER REWARD PERSONALIZAT ---
class RacingRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        vehicle = self.env.unwrapped.vehicle
        config = self.env.unwrapped.config
        
        speed = vehicle.speed
        _, lateral = vehicle.lane.local_coordinates(vehicle.position)
        
        if speed > 0: speed_reward = speed / config["speed_limit"]
        else: speed_reward = -0.5
            
        lane_centering_reward = 1 / (1 + config["lane_centering_cost"] * lateral**2)
        
        steering_stability_reward = -(action[1]**2)
        
        # === CALCUL FINAL ===
        weighted_reward = (
            0.6 * speed_reward +
            0.3 * lane_centering_reward +
            0.1 * steering_stability_reward
        )
        
        if vehicle.crashed: weighted_reward = -1.0
        
        final_reward = weighted_reward * float(vehicle.on_road)
        
        return obs, final_reward, done, truncated, info

def train_local_expert():
    print(f">>> [1/2] Antrenăm expertul RACER (PPO)...")
    
    env = make_vec_env(
        ENV_NAME, 
        n_envs=N_CPU, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"config": CONFIG_RACING},
        wrapper_class=RacingRewardWrapper
    )
    
    model = PPO("MlpPolicy", env, verbose=1, device=DEVICE, batch_size=256, n_steps=1024, learning_rate=5e-4)
    model.learn(total_timesteps=60000) 
    
    print(">>> Expert Racer antrenat! Îl salvăm.")
    model.save("expert_local")
    env.close()
    return model

def record_data(model, filename_out, randomness_level=0.0):
    print(f"\n--> Generăm dataset: {filename_out} (Err: {randomness_level*100}%)")
    
    raw_env = gym.make(ENV_NAME, render_mode=None, config=CONFIG_RACING)
    env = RacingRewardWrapper(raw_env)
    
    observations = []
    actions = []
    
    obs, _ = env.reset()
    count = 0
    max_steps = 3000
    
    while count < max_steps:
        if np.random.rand() > randomness_level:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample() 

        next_obs, reward, done, truncated, _ = env.step(action)
        
        observations.append(obs)
        actions.append(action)
        
        obs = next_obs
        count += 1
        
        if done or truncated:
            obs, _ = env.reset()
            
    dataset = {"observations": np.array(observations), "actions": np.array(actions)}
    with open(filename_out, "wb") as f:
        pickle.dump(dataset, f)
    env.close()

if __name__ == "__main__":
    torch.set_num_threads(1)
    
    expert_model = train_local_expert()
    
    record_data(expert_model, "dataset_slab.pkl", randomness_level=1.0)
    record_data(expert_model, "dataset_mediu.pkl", randomness_level=0.3)
    record_data(expert_model, "dataset_expert.pkl", randomness_level=0.0)
    
    print("\n=== DATE GENERATE (MOD RACING) ===")
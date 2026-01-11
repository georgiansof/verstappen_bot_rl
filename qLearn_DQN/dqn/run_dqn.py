import gymnasium as gym
import highway_env
import torch
import numpy as np
import time
import json
import os
import sys
from gymnasium.envs.registration import register

# Register
register(
    id='first-custom-map',
    entry_point='dawd.maps.first_custom_map:FirstCustomMap',
)

register(
    id='elipse',
    entry_point='dawd.maps.elipse:ElipseMap',
)

register(
    id='formula1',
    entry_point='dawd.maps.formula_one_track:FormulaOneMap',
)

register(
    id='moto',
    entry_point='dawd.maps.moto_gp:MotoGpMap',
)

# Import the Agent and Network classes from your training file
# !!! Make sure your training script is saved as 'dqn_training.py' !!!
try:
    from fInal.dqn.dgn import DQNAgent
    from fInal.dqn.dgn import RacingRewardWrapper
except ImportError:
    print("Error: Could not import DQNAgent.")
    print("Please save your training code as 'dqn_training.py' in this folder.")
    sys.exit(1)

def run_visualization(model_path, save_dir='results_dqn', episodes=5):
   
        
    env = gym.make('racetrack-v0', render_mode='human')
    env.unwrapped.configure({
        "observation": {
            "type": "OccupancyGrid",
            "features": ['presence', 'on_road'],
            "grid_size": [[-18, 18], [-18, 18]],
            "grid_step": [3, 3],
            "as_image": False,
            "align_to_vehicle_axes": True
        },
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
            "acceleration_range": [-4, 4],
            "steering_range": [-0.6, 0.6] # Redus puțin pentru stabilitate
        }
    })
    env.reset()
    env = RacingRewardWrapper(env)
    print(env.action_space)
    print(f"Action space: {env.action_space}")
    print(f"Action space type: {type(env.action_space)}")

    # Dacă e Discrete:
    if isinstance(env.action_space, gym.spaces.Discrete):
        print(f"Number of actions: {env.action_space.n}")
    # Dacă e Box:
    elif isinstance(env.action_space, gym.spaces.Box):
        print(f"Action shape: {env.action_space.shape}")
    
    
    state_dim = env.observation_space.shape[0]
    action_space = env.action_space

    
    EPISODES = 1000
    LR = 1e-4
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    BUFFER_SIZE = 4000
    BATCH_SIZE = 64
    TARGET_UPDATE = 10
    ACTION_BINS = 5
    
    # Creăm agentul
    agent = DQNAgent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=LR,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE,
        action_bins=ACTION_BINS
    )
    # 4. Load Trained Weights
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        agent.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Set network to evaluation mode (disables Dropout/BatchNorm if used)
    agent.policy_net.eval()

    # 5. Simulation Loop
    print(f"\nRunning {episodes} episodes...")
    print("Press Ctrl+C to stop.")

    try:
        for episode in range(episodes):
            state, info = env.reset()
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            
            print(f"Starting Episode {episode + 1}...")
            
            while not (done or truncated):
                # Handle window closing
                env.render()
                
                # Get Action (Deterministic/Greedy because training=False)
                action, _ = agent.get_action(state, training=False)
                
                # Step
                state, reward, done, truncated, info = env.step(action)
                
                total_reward += reward
                steps += 1
                
                # Optional: Slow down visualization slightly
                # time.sleep(0.01) 
            
            print(f"Episode {episode + 1} Finished: Reward = {total_reward:.2f}, Steps = {steps}")
            time.sleep(0.5) # Pause between episodes

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    # Path to your saved model
    MODEL_PATH = "final_model.pt"
    # Or use a specific checkpoint:
    # MODEL_PATH = "results_dqn/checkpoint_ep900.pt"
    
    run_visualization(MODEL_PATH)
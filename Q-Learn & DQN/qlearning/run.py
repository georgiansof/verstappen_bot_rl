import gymnasium as gym
import highway_env
import numpy as np
import pickle
from fInal.qlearning.q_learning2 import QLearningAgent  # import your agent class from your main code
import pickle
from gymnasium.envs.registration import register

register(
    id='moto',
    entry_point='dawd.maps.moto_gp:MotoGpMap',
)

register(
    id='first-custom-map',
    entry_point='dawd.maps.first_custom_map:FirstCustomMap',
)


def load_agent(checkpoint_path, action_space, state_bins, action_bins=3):
    """Load Q-table and return a QLearningAgent"""
    agent = QLearningAgent(
        action_space=action_space,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=0.0,  # greedy
        epsilon_end=0.0,
        epsilon_decay=1.0,
        state_bins=state_bins,
        action_bins=action_bins
    )

    data = pickle.load(open(checkpoint_path, 'rb'))

    if isinstance(data, dict) and 'q_table' in data:
        q_table = data['q_table']
    else:
        # probably a direct q_table
        q_table = data

    print(f"Loaded Q-table with {len(q_table)} states")
    agent.q_table = q_table
 
    return agent

def run_agent(env, agent, n_episodes=5, max_steps=1000, render_mode='human'):
    """Run the agent in the environment"""
    env = gym.make(env.spec.id, render_mode=render_mode) if hasattr(env, 'spec') else env
    
    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not (done or truncated) and steps < max_steps:
            action, _ = agent.get_action(state, training=False)
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            env.render()
        
        print(f"Episode {episode+1}: Reward = {total_reward:.2f}, Steps = {steps}")
    
    env.close()

if __name__ == "__main__":
    env = gym.make('racetrack-v0', render_mode='human')

    checkpoint_path = 'final_q_table.pkl'
    
    state_bins = [8, 4, 5]

    agent = load_agent(checkpoint_path, env.action_space, state_bins, action_bins=3)
    
    run_agent(env, agent, n_episodes=10)

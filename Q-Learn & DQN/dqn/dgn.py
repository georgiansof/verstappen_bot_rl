import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime
import os
from gymnasium.envs.registration import register
register(
    id='mgigi',
    entry_point='dawd.maps.muiegigi:ElipseOvertakingMap',
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """Experience Replay Buffer pentru DQN"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        """Adaugă o tranziție în buffer"""
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        """Sample un batch random din buffer"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQN_CNN(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super(DQN_CNN, self).__init__()
        channels, height, width = obs_shape
        
        # Două straturi sunt suficiente pentru 12x12
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Flatten size: 64 * 12 * 12 = 9216
        self.fc1 = nn.Linear(64 * height * width, 512)
        self.fc2 = nn.Linear(512, action_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class RacingRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        vehicle = self.env.unwrapped.vehicle
        config = self.env.unwrapped.config
        
        # 1. Ne asigurăm că speed este scalar
        speed = float(np.mean(vehicle.speed))
        
        # 2. Calculăm lane centering
        _, lateral = vehicle.lane.local_coordinates(vehicle.position)
        # Folosim .get() ca să nu crape dacă lipsește din config
        lc_cost = config.get("lane_centering_cost", 4)
        lane_centering_reward = 1 / (1 + lc_cost * (float(lateral)**2))
        
        # 3. Reward viteză (cu speed_limit asigurat)
        limit = config.get("speed_limit", 20)
        speed_reward = speed / limit if speed > 0 else -0.5
            
        # 4. Stabilitate (action[1] este steering)
        steering_stability_reward = -(float(action[1])**2)
        
        # 5. Calcul weighted
        weighted_reward = (
            0.6 * speed_reward +
            0.3 * lane_centering_reward +
            0.1 * steering_stability_reward
        )
        
        if vehicle.crashed: 
            weighted_reward = -1.0
        
        # REZOLVARE EROARE: Verificăm dacă on_road este array sau scalar
        on_road_val = np.all(vehicle.on_road) # True doar dacă toate punctele sunt pe drum
        final_reward = float(weighted_reward) * float(on_road_val)
        
        return obs, final_reward, done, truncated, info


class DQNAgent:
    """Agent DQN pentru highway-env cu acțiuni continue discretizate"""
    
    def __init__(self, obs_space, action_space, 
                 lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, 
                 epsilon_decay=0.995, buffer_size=100000, batch_size=64,
                 target_update_freq=10, action_bins=5, use_cnn=True):
        """
        Args:
            obs_space: spațiul de observații
            action_space: spațiul de acțiuni (Box)
            use_cnn: True pentru CNN (grid obs), False pentru MLP (flatten obs)
        """
        self.obs_space = obs_space
        self.action_space = action_space
        self.action_dim = action_space.shape[0]
        self.action_bins = action_bins
        self.use_cnn = use_cnn
        
        # Creăm acțiuni discrete
        self.discrete_actions = self._create_discrete_actions()
        self.n_actions = len(self.discrete_actions)
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        print(f"Folosim CNN pentru observații grid: {obs_space.shape}")
        self.policy_net = DQN_CNN(obs_space.shape, self.n_actions).to(device)
        self.target_net = DQN_CNN(obs_space.shape, self.n_actions).to(device)
        self.use_cnn = True
       
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.update_count = 0
        
        print(f"DQN Agent creat cu {self.n_actions} acțiuni discrete")
        print(f"Parameters: {sum(p.numel() for p in self.policy_net.parameters()):,}")
    
        
    def _create_discrete_actions(self):
        """Creează set discret de acțiuni din spațiul continuu"""
        discrete_actions = []
        action_values = []
        
        for i in range(self.action_dim):
            low = self.action_space.low[i]
            high = self.action_space.high[i]
            values = np.linspace(low, high, self.action_bins)
            action_values.append(values)
        
        if self.action_dim == 1:
            discrete_actions = [[v] for v in action_values[0]]
        elif self.action_dim == 2:
            for acc in action_values[0]:
                for steer in action_values[1]:
                    discrete_actions.append([acc, steer])
        else:
            import itertools
            for combo in itertools.product(*action_values):
                discrete_actions.append(list(combo))
        
        return [np.array(a, dtype=np.float32) for a in discrete_actions]
    
    def _preprocess_state(self, state):
        return state.astype(np.float32)

    
    def get_action(self, state, training=True):
        """Selectează acțiune folosind epsilon-greedy"""
        if training and random.random() < self.epsilon:
            # Explorare
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            # Exploatare
            with torch.no_grad():
                state_processed = self._preprocess_state(state)
                state_tensor = torch.FloatTensor(state_processed).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.argmax(1).item()
        
        return self.discrete_actions[action_idx], action_idx
    
    def store_transition(self, state, action_idx, reward, next_state, done):
        """Stochează tranziția în replay buffer"""
        # Preprocesăm stările înainte de stocare
        state_processed = self._preprocess_state(state)
        next_state_processed = self._preprocess_state(next_state)
        self.replay_buffer.push(state_processed, action_idx, reward, next_state_processed, done)
    
    def update(self):
        """Update network folosind batch din replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert la tensori
        state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(device)
        done_batch = torch.FloatTensor(batch.done).to(device)
        
        # Compute Q(s_t, a)
        current_q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        
        # Compute V(s_{t+1})
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        self.update_count += 1
        
        # Update target network
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon după fiecare episod"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint['update_count']


def train_dqn(env, agent, n_episodes=1000, save_freq=100, save_dir='results_dqn'):
    """Antrenează agentul DQN"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Metrici
    episode_rewards = []
    episode_lengths = []
    epsilon_history = []
    loss_history = []
    q_value_history = []
    
    print(f"\nÎncepem antrenamentul DQN pentru {n_episodes} episoade...")
    print(f"Device: {device}")
    print(f"Batch size: {agent.batch_size}, Buffer capacity: {agent.replay_buffer.buffer.maxlen}")
    print("-" * 70)
    
    for episode in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        episode_q_values = []
        done = False
        truncated = False
        
        while not (done or truncated):
            action, action_idx = agent.get_action(state, training=True)
            
            next_state, reward, done, truncated, info = env.step(action)

            agent.store_transition(state, action_idx, reward, next_state, done or truncated)

            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)

            if episode_length % 10 == 0 and len(agent.replay_buffer) >= agent.batch_size:
                with torch.no_grad():
                    state_processed = agent._preprocess_state(state)
                    state_tensor = torch.FloatTensor(state_processed).unsqueeze(0).to(device)
                    q_vals = agent.policy_net(state_tensor)
                    episode_q_values.append(q_vals.max().item())
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if episode_length > 1000:
                break
        
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        epsilon_history.append(agent.epsilon)
        
        if episode_losses:
            loss_history.append(np.mean(episode_losses))
        else:
            loss_history.append(0)
            
        if episode_q_values:
            q_value_history.append(np.mean(episode_q_values))
        else:
            q_value_history.append(0)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            avg_loss = np.mean([l for l in loss_history[-50:] if l > 0]) if any(loss_history[-50:]) else 0
            avg_q = np.mean([q for q in q_value_history[-50:] if q > 0]) if any(q_value_history[-50:]) else 0
            buffer_size = len(agent.replay_buffer)
            
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Reward: {avg_reward:.2f} | "
                  f"Length: {avg_length:.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Q-val: {avg_q:.2f} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Buffer: {buffer_size}")
        
        # Salvare checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_ep{episode + 1}.pt')
            agent.save(checkpoint_path)
            print(f"  → Checkpoint salvat: {checkpoint_path}")
    
    print("-" * 70)
    print("Antrenament finalizat!")
    
    # Salvăm datele
    training_data = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'epsilon_history': epsilon_history,
        'loss_history': loss_history,
        'q_value_history': q_value_history,
        'hyperparams': {
            'lr': agent.optimizer.param_groups[0]['lr'],
            'gamma': agent.gamma,
            'epsilon_start': 1.0,
            'epsilon_end': agent.epsilon_end,
            'epsilon_decay': agent.epsilon_decay,
            'batch_size': agent.batch_size,
            'buffer_size': agent.replay_buffer.buffer.maxlen,
            'target_update_freq': agent.target_update_freq,
            'action_bins': agent.action_bins,
            'n_discrete_actions': agent.n_actions,
            'n_episodes': n_episodes,
            'use_cnn': agent.use_cnn
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'device': str(device)
    }
    
    with open(os.path.join(save_dir, 'training_data.json'), 'w') as f:
        json.dump(training_data, f, indent=2)
    
    # Salvare model final
    agent.save(os.path.join(save_dir, 'final_model.pt'))
    
    print(f"Date salvate în: {save_dir}/")
    
    return training_data


def plot_results(training_data, save_dir='results_dqn'):
    """Generează grafice pentru DQN"""
    
    episode_rewards = training_data['episode_rewards']
    episode_lengths = training_data['episode_lengths']
    epsilon_history = training_data['epsilon_history']
    loss_history = training_data['loss_history']
    q_value_history = training_data['q_value_history']
    
    window = 50
    rewards_ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    lengths_ma = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Episode Rewards
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(episode_rewards, alpha=0.3, label='Raw', color='blue')
    ax1.plot(range(window-1, len(episode_rewards)), rewards_ma, 
             label=f'MA({window})', color='red', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode Lengths
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(episode_lengths, alpha=0.3, label='Raw', color='green')
    ax2.plot(range(window-1, len(episode_lengths)), lengths_ma, 
             label=f'MA({window})', color='orange', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Lengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Loss Evolution
    ax3 = plt.subplot(3, 3, 3)
    loss_nonzero = [l for l in loss_history if l > 0]
    if loss_nonzero:
        ax3.plot(loss_nonzero, alpha=0.5, color='red')
        if len(loss_nonzero) > window:
            loss_ma = np.convolve(loss_nonzero, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(loss_nonzero)), loss_ma, 
                     color='darkred', linewidth=2, label=f'MA({window})')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss (Huber)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Values Evolution
    ax4 = plt.subplot(3, 3, 4)
    q_nonzero = [q for q in q_value_history if q != 0]
    if q_nonzero:
        ax4.plot(q_nonzero, alpha=0.5, color='purple')
        if len(q_nonzero) > window:
            q_ma = np.convolve(q_nonzero, np.ones(window)/window, mode='valid')
            ax4.plot(range(window-1, len(q_nonzero)), q_ma, 
                     color='indigo', linewidth=2, label=f'MA({window})')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Max Q-Value')
    ax4.set_title('Q-Value Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Epsilon Decay
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(epsilon_history, color='teal', linewidth=2)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Epsilon')
    ax5.set_title('Epsilon Decay (Exploration)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Reward Distribution
    ax6 = plt.subplot(3, 3, 6)
    last_rewards = episode_rewards[-200:]
    ax6.hist(last_rewards, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax6.axvline(np.mean(last_rewards), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(last_rewards):.2f}')
    ax6.set_xlabel('Reward')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Reward Distribution (Last 200)')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Learning Curve (Reward vs Loss)
    ax7 = plt.subplot(3, 3, 7)
    if loss_nonzero:
        min_len = min(len(rewards_ma), len(loss_nonzero))
        ax7_twin = ax7.twinx()
        ax7.plot(range(window-1, window-1+min_len), rewards_ma[:min_len], 
                 color='blue', label='Reward', linewidth=2)
        ax7_twin.plot(loss_nonzero[:min_len], color='red', 
                      label='Loss', linewidth=2, alpha=0.7)
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('Reward', color='blue')
        ax7_twin.set_ylabel('Loss', color='red')
        ax7.set_title('Learning Curve')
        ax7.grid(True, alpha=0.3)
    
    # 8. Performance Improvement
    ax8 = plt.subplot(3, 3, 8)
    n = len(episode_rewards)
    quartile_size = n // 4
    quartiles = [
        np.mean(episode_rewards[i*quartile_size:(i+1)*quartile_size]) 
        for i in range(4)
    ]
    ax8.bar(['Q1', 'Q2', 'Q3', 'Q4'], quartiles, color=['red', 'orange', 'yellow', 'green'])
    ax8.set_ylabel('Mean Reward')
    ax8.set_title('Performance by Quarter')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Summary Stats
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    final_100_reward = np.mean(episode_rewards[-100:])
    final_100_length = np.mean(episode_lengths[-100:])
    final_loss = np.mean([l for l in loss_history[-100:] if l > 0]) if any(loss_history[-100:]) else 0
    final_q = np.mean([q for q in q_value_history[-100:] if q > 0]) if any(q_value_history[-100:]) else 0
    
    summary_text = f"""
    DQN FINAL STATISTICS (Last 100 Episodes)
    
    Mean Reward: {final_100_reward:.2f}
    Mean Length: {final_100_length:.1f}
    Mean Loss: {final_loss:.4f}
    Mean Q-Value: {final_q:.2f}
    
    Final Epsilon: {epsilon_history[-1]:.4f}
    
    Hyperparameters:
    - Learning Rate: {training_data['hyperparams']['lr']}
    - Gamma: {training_data['hyperparams']['gamma']}
    - Batch Size: {training_data['hyperparams']['batch_size']}
    - Buffer Size: {training_data['hyperparams']['buffer_size']}
    - Target Update: {training_data['hyperparams']['target_update_freq']}
    - Action Bins: {training_data['hyperparams']['action_bins']}
    - Use CNN: {training_data['hyperparams']['use_cnn']}
    
    Device: {training_data['device']}
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center')
    
    plt.suptitle('DQN Training Results - Highway RaceTrack', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'training_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Grafice salvate în: {plot_path}")
    plt.close()


def evaluate_agent(env, agent, n_episodes=20):
    """Evaluează agentul DQN"""
    
    print(f"\nEvaluare agent pe {n_episodes} episoade...")
    
    eval_rewards = []
    eval_lengths = []
    
    for episode in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _ = agent.get_action(state, training=False)
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if episode_length > 1000:
                break
        
        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
        
        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    print(f"\nRezultate evaluare:")
    print(f"  Mean Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"  Mean Length: {np.mean(eval_lengths):.1f} ± {np.std(eval_lengths):.1f}")
    print(f"  Max Reward: {np.max(eval_rewards):.2f}")
    print(f"  Min Reward: {np.min(eval_rewards):.2f}")
    
    return {
        'mean_reward': float(np.mean(eval_rewards)),
        'std_reward': float(np.std(eval_rewards)),
        'mean_length': float(np.mean(eval_lengths)),
        'max_reward': float(np.max(eval_rewards)),
        'min_reward': float(np.min(eval_rewards)),
        'all_rewards': eval_rewards,
        'all_lengths': eval_lengths
    }


def main():
    """Funcția principală pentru DQN"""
    
    # Environment
    env = gym.make('racetrack-v0')


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
        },
        "speed_limit": 20,
        "lane_centering_cost": 4,
    })
    env.reset()
    env = RacingRewardWrapper(env)
    
    # 4. Folosim env.unwrapped pentru a ne asigura că luăm noile dimensiuni (2, 12, 12) și (2,)
    obs_space = env.unwrapped.observation_space
    act_space = env.unwrapped.action_space
        
    print("=" * 70)
    print("DQN (DEEP Q-NETWORK) - HIGHWAY-ENV RACETRACK")
    print("=" * 70)
    print(f"Environment: racetrack-v0")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print("=" * 70)
    
    # Hyperparametri
    EPISODES = 1000
    LR = 1e-4
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    TARGET_UPDATE = 10
    ACTION_BINS = 5
    
    # Creăm agentul
    agent = DQNAgent(
        obs_space=obs_space,
        action_space=act_space,
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
    
    # Antrenare
    training_data = train_dqn(
        env=env,
        agent=agent,
        n_episodes=EPISODES,
        save_freq=100,
        save_dir='results_dqn'
    )
    
    # Grafice
    plot_results(training_data, save_dir='results_dqn')
    
    # Evaluare
    eval_results = evaluate_agent(env, agent, n_episodes=20)
    
    with open('results_dqn/evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("ANTRENAMENT DQN FINALIZAT!")
    print(f"Rezultate salvate în: results_dqn/")
    print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    main()
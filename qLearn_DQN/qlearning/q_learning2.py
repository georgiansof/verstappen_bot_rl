import gymnasium as gym
import highway_env
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Fix pentru matplotlib warnings
import matplotlib.pyplot as plt
import pickle
import json
from collections import defaultdict
from datetime import datetime
import os

class QLearningAgent:
    """Agent Q-Learning tabular pentru highway-env racetrack cu acțiuni continue"""
    
    def __init__(self, action_space, alpha=0.1, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 state_bins=[8, 8, 4, 4, 4], action_bins=5):
        self.action_space = action_space
        self.action_dim = action_space.shape[0]
        self.action_bins = action_bins
        
        # Creăm un set discret de acțiuni
        self.discrete_actions = self._create_discrete_actions()
        self.n_actions = len(self.discrete_actions)
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.state_bins = state_bins
        self.prev_pos = None
        
        # Q-table implementată ca dicționar (sparse)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        
        print(f"Agent creat cu {self.n_actions} acțiuni discrete")
        print(f"Action space: low={action_space.low}, high={action_space.high}")
        
    def _create_discrete_actions(self):
        """
        Creează un set discret de acțiuni din spațiul continuu.
        Pentru racetrack: acțiuni [acceleration, steering]
        """
        discrete_actions = []
        
        # Pentru fiecare dimensiune de acțiune, creăm bins uniforme
        action_values = []
        for i in range(self.action_dim):
            low = self.action_space.low[i]
            high = self.action_space.high[i]
            values = [-1.0, 0.0, 1.0]
            action_values.append(values)
        
        # Combinații de acțiuni (produs cartezian)
        if self.action_dim == 1:
            discrete_actions = [[v] for v in action_values[0]]
        elif self.action_dim == 2:
            for acc in action_values[0]:
                for steer in action_values[1]:
                    discrete_actions.append([acc, steer])
        else:
            # Generalizare pentru mai multe dimensiuni
            import itertools
            for combo in itertools.product(*action_values):
                discrete_actions.append(list(combo))
        
        return [np.array(a) for a in discrete_actions]
    
    def discretize_state(self, observation):
        road = observation[0]
        agent = observation[1]

        H, W = agent.shape
        ys, xs = np.where(agent > 0)

        if len(xs) == 0:
            x = y = 0
        else:
            x = int(xs.mean())
            y = int(ys.mean())

        # road geometry
        front = road[max(y-2, 0):y, x].mean() if y > 0 else 0.0
        left  = road[y, max(x-1, 0):x].mean() if x > 0 else 0.0
        right = road[y, x+1:min(x+2, W)].mean() if x < W-1 else 0.0

        features = [
            x / W,
            front,
            left - right
        ]

        bins = [8, 4, 5]
        return tuple(int(np.clip(f, -1, 1) * (b - 1) / 2) for f, b in zip(features, bins))

    def get_action(self, state, training=True):
        """Selectează acțiune folosind epsilon-greedy"""
        discrete_state = self.discretize_state(state)  # ✅ Calculează starea PRIMUL
        
        # ✅ Verifică dacă starea există ÎNAINTE de a o accesa
        if discrete_state not in self.q_table:
            # Stare nouă, necunoscută - returnează acțiune neutră
            action_idx = self.n_actions // 2  # Mijloc (0 pentru 3 acțiuni: [-1, 0, 1])
            return self.discrete_actions[action_idx], action_idx
        
        if training and np.random.random() < self.epsilon:
            # Explorare: acțiune aleatorie
            action_idx = np.random.randint(self.n_actions)
        else:
            # Exploatare: best Q-value
            action_idx = np.argmax(self.q_table[discrete_state])
        
        return self.discrete_actions[action_idx], action_idx
    
    def update(self, state, action_idx, reward, next_state, done):
        """Update Q-value folosind Q-Learning formula"""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Q-Learning update
        current_q = self.q_table[discrete_state][action_idx]
        
        if done:
            td_target = reward
        else:
            max_next_q = np.max(self.q_table[discrete_next_state])
            td_target = reward + self.gamma * max_next_q
        
        td_error = td_target - current_q
        self.q_table[discrete_state][action_idx] = current_q + self.alpha * td_error
        
        return td_error
    
    def decay_epsilon(self):
        """Decay epsilon după fiecare episod"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_q_table_stats(self):
        """Returnează statistici despre Q-table"""
        if len(self.q_table) == 0:
            return {
                'n_states': 0,
                'mean_q': 0,
                'max_q': 0,
                'min_q': 0,
                'std_q': 0
            }
        
        all_q_values = []
        for state_q_values in self.q_table.values():
            all_q_values.extend(state_q_values)
        
        all_q_values = np.array(all_q_values)
        
        return {
            'n_states': len(self.q_table),
            'mean_q': float(np.mean(all_q_values)),
            'max_q': float(np.max(all_q_values)),
            'min_q': float(np.min(all_q_values)),
            'std_q': float(np.std(all_q_values))
        }


def train_qlearning(env, agent, n_episodes=1000, save_freq=100, save_dir='results_qlearning'):
    """
    Antrenează agentul Q-Learning
    """
    agent.prev_pos = None
    os.makedirs(save_dir, exist_ok=True)
    
    # Metrici de tracking
    episode_rewards = []
    episode_lengths = []
    epsilon_history = []
    q_stats_history = []
    td_errors_history = []
    
    print(f"Începem antrenamentul Q-Learning pentru {n_episodes} episoade...")
    print(f"Parametri: alpha={agent.alpha}, gamma={agent.gamma}, epsilon={agent.epsilon:.3f}")
    print(f"State bins: {agent.state_bins}, Action bins: {agent.action_bins}")
    print(f"Total discrete actions: {agent.n_actions}")
    print("-" * 70)
    
    for episode in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_td_errors = []
        done = False
        truncated = False
        
        while not (done or truncated):
            # Selectăm acțiune
            action, action_idx = agent.get_action(state, training=True)
            
            # Executăm acțiunea
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update Q-table
            td_error = agent.update(state, action_idx, reward, next_state, done or truncated)
            episode_td_errors.append(abs(td_error))
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            # Limită de siguranță
            if episode_length > 1000:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Salvăm metrici
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        epsilon_history.append(agent.epsilon)
        if episode_td_errors:
            td_errors_history.append(np.mean(episode_td_errors))
        
        # Q-table stats la fiecare 10 episoade
        if episode % 10 == 0:
            q_stats = agent.get_q_table_stats()
            q_stats['episode'] = episode
            q_stats_history.append(q_stats)
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            avg_td_error = np.mean(td_errors_history[-50:]) if td_errors_history else 0
            q_stats = agent.get_q_table_stats()
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"TD Error: {avg_td_error:.4f} | "
                  f"Q-States: {q_stats['n_states']}")
        
        # Salvare checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_ep{episode + 1}.pkl')
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'q_table': dict(agent.q_table),
                    'episode': episode + 1,
                    'epsilon': agent.epsilon,
                    'hyperparams': {
                        'alpha': agent.alpha,
                        'gamma': agent.gamma,
                        'epsilon_decay': agent.epsilon_decay,
                        'state_bins': agent.state_bins,
                        'action_bins': agent.action_bins
                    }
                }, f)
            print(f"  → Checkpoint salvat: {checkpoint_path}")
    
    print("-" * 70)
    print("Antrenament finalizat!")
    
    # Salvăm toate datele
    training_data = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'epsilon_history': epsilon_history,
        'td_errors_history': td_errors_history,
        'q_stats_history': q_stats_history,
        'hyperparams': {
            'alpha': agent.alpha,
            'gamma': agent.gamma,
            'epsilon_start': 1.0,
            'epsilon_end': agent.epsilon_end,
            'epsilon_decay': agent.epsilon_decay,
            'state_bins': agent.state_bins,
            'action_bins': agent.action_bins,
            'n_discrete_actions': agent.n_actions,
            'n_episodes': n_episodes
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Salvare date training
    with open(os.path.join(save_dir, 'training_data.json'), 'w') as f:
        json.dump(training_data, f, indent=2)
    
    # Salvare Q-table final
    with open(os.path.join(save_dir, 'final_q_table.pkl'), 'wb') as f:
        pickle.dump(dict(agent.q_table), f)
    
    print(f"Date salvate în: {save_dir}/")
    
    return training_data


def plot_results(training_data, save_dir='results_qlearning'):
    """Generează grafice pentru rezultatele antrenamentului"""
    
    episode_rewards = training_data['episode_rewards']
    episode_lengths = training_data['episode_lengths']
    epsilon_history = training_data['epsilon_history']
    td_errors_history = training_data.get('td_errors_history', [])
    q_stats_history = training_data['q_stats_history']
    
    # Calculăm moving average
    window = 50
    rewards_ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    lengths_ma = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
    
    # Creăm figura cu subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Episode Rewards
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(episode_rewards, alpha=0.3, label='Raw', color='blue')
    ax1.plot(range(window-1, len(episode_rewards)), rewards_ma, label=f'MA({window})', color='red', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode Lengths
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(episode_lengths, alpha=0.3, label='Raw', color='green')
    ax2.plot(range(window-1, len(episode_lengths)), lengths_ma, label=f'MA({window})', color='orange', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Lengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Epsilon Decay
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(epsilon_history, color='purple', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Epsilon Decay (Exploration Rate)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-table Size Growth
    ax4 = plt.subplot(3, 3, 4)
    episodes_q = [stat['episode'] for stat in q_stats_history]
    n_states = [stat['n_states'] for stat in q_stats_history]
    ax4.plot(episodes_q, n_states, color='teal', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Number of States')
    ax4.set_title('Q-Table Size Growth')
    ax4.grid(True, alpha=0.3)
    
    # 5. Q-Values Statistics
    ax5 = plt.subplot(3, 3, 5)
    mean_q = [stat['mean_q'] for stat in q_stats_history]
    max_q = [stat['max_q'] for stat in q_stats_history]
    min_q = [stat['min_q'] for stat in q_stats_history]
    ax5.plot(episodes_q, mean_q, label='Mean Q', linewidth=2)
    ax5.plot(episodes_q, max_q, label='Max Q', linewidth=2, alpha=0.7)
    ax5.plot(episodes_q, min_q, label='Min Q', linewidth=2, alpha=0.7)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Q-Value')
    ax5.set_title('Q-Values Evolution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. TD Errors
    if td_errors_history:
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(td_errors_history, color='red', alpha=0.5)
        if len(td_errors_history) > window:
            td_ma = np.convolve(td_errors_history, np.ones(window)/window, mode='valid')
            ax6.plot(range(window-1, len(td_errors_history)), td_ma, color='darkred', linewidth=2, label=f'MA({window})')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('TD Error')
        ax6.set_title('TD Error Evolution')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    # 7. Reward Distribution (last 200 episodes)
    ax7 = plt.subplot(3, 3, 7)
    last_rewards = episode_rewards[-200:]
    ax7.hist(last_rewards, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax7.axvline(np.mean(last_rewards), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(last_rewards):.2f}')
    ax7.set_xlabel('Reward')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Reward Distribution (Last 200 Episodes)')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Q-Value Standard Deviation
    ax8 = plt.subplot(3, 3, 8)
    std_q = [stat['std_q'] for stat in q_stats_history]
    ax8.plot(episodes_q, std_q, color='purple', linewidth=2)
    ax8.set_xlabel('Episode')
    ax8.set_ylabel('Std Q-Value')
    ax8.set_title('Q-Value Variability')
    ax8.grid(True, alpha=0.3)
    
    # 9. Learning Progress Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Statistici finale
    final_100_reward = np.mean(episode_rewards[-100:])
    final_100_length = np.mean(episode_lengths[-100:])
    final_q_stats = q_stats_history[-1] if q_stats_history else {}
    
    summary_text = f"""
    FINAL STATISTICS (Last 100 Episodes)
    
    Mean Reward: {final_100_reward:.2f}
    Mean Length: {final_100_length:.1f}
    
    Q-Table Size: {final_q_stats.get('n_states', 0)}
    Mean Q-Value: {final_q_stats.get('mean_q', 0):.3f}
    Max Q-Value: {final_q_stats.get('max_q', 0):.3f}
    
    Final Epsilon: {epsilon_history[-1]:.4f}
    
    Hyperparameters:
    - Alpha: {training_data['hyperparams']['alpha']}
    - Gamma: {training_data['hyperparams']['gamma']}
    - State Bins: {training_data['hyperparams']['state_bins']}
    - Action Bins: {training_data['hyperparams']['action_bins']}
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.suptitle('Q-Learning Training Results - Highway RaceTrack', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Salvare
    plot_path = os.path.join(save_dir, 'training_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Grafice salvate în: {plot_path}")
    plt.close()


def evaluate_agent(env, agent, n_episodes=10):
    """Evaluează agentul antrenat"""
    
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
            # Folosim politica greedy (fără explorare)
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
    """Funcția principală pentru antrenament Q-Learning"""
    
    # Configurare environment
    env = gym.make('racetrack-v0')
    
    print("=" * 70)
    print("Q-LEARNING TABULAR - HIGHWAY-ENV RACETRACK")
    print("=" * 70)
    print(f"Environment: racetrack-v0")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Action space low: {env.action_space.low}")
    print(f"Action space high: {env.action_space.high}")
    print("=" * 70)
    
    # Hyperparametri
    EPISODES = 1500
    ALPHA = 0.1
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    STATE_BINS = [8, 4, 5]
    ACTION_BINS = 3
    EPSILON_DECAY = 0.999
    
    agent = QLearningAgent(
        action_space=env.action_space,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        state_bins=STATE_BINS,
        action_bins=ACTION_BINS
    )
    
    # Antrenare
    training_data = train_qlearning(
        env=env,
        agent=agent,
        n_episodes=EPISODES,
        save_freq=100,
        save_dir='results_qlearning'
    )
    
    plot_results(training_data, save_dir='results_qlearning')
    
    eval_results = evaluate_agent(env, agent, n_episodes=20)
    
    with open('results_qlearning/evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("ANTRENAMENT FINALIZAT!")
    print(f"Rezultate salvate în: results_qlearning/")
    print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    main()
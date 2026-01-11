import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
import argparse

ENV_NAME = "racetrack-v0"
ALGO = PPO

# Configurație pentru control lateral și longitudinal complet
ENV_CONFIG = {
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,  # Accelerație/Frână
        "lateral": True,  # Volan (Steering)
        "acceleration_range": [-4, 4],
        "steering_range": [-0.78, 0.78]  # Aproximativ 45 de grade în radiani
    },
    "speed_limit": 15.0,  # Creștem limita de viteză pentru racing
    "other_vehicles": 5,  # Mai mult trafic!
}


class RacingRewardWrapper(gym.Wrapper):
    """
    Wrapper care modifică recompensele pentru a prioritiza VITEZA.
    - 60% importanță pe viteză
    - 30% pe centrare (permite tăierea curbelor)
    - 10% pe stabilitate (evită vibratul volanului)
    """

    def __init__(self, env, speed_weight=0.6, centering_weight=0.3, stability_weight=0.1):
        super().__init__(env)
        self.speed_weight = speed_weight
        self.centering_weight = centering_weight
        self.stability_weight = stability_weight
        self.speed_limit = env.unwrapped.config.get("speed_limit", 10.0)
        self.lane_centering_cost = env.unwrapped.config.get("lane_centering_cost", 4)

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        vehicle = self.env.unwrapped.vehicle

        # 1. SPEED REWARD - cea mai importantă pentru racing
        speed = vehicle.speed
        if speed > 0:
            speed_reward = min(speed / self.speed_limit, 1.5)  # Bonus pentru depășirea limitei
        else:
            speed_reward = -0.5  # Penalizare pentru mers înapoi

        # 2. LANE CENTERING - permite tăierea curbelor
        if vehicle.lane is not None:
            _, lateral = vehicle.lane.local_coordinates(vehicle.position)
            lane_centering_reward = 1 / (1 + self.lane_centering_cost * lateral ** 2)
        else:
            lane_centering_reward = 0

        # 3. STEERING STABILITY - evită vibratul volanului
        steering_action = action[1] if len(action) > 1 else action[0]
        steering_stability = -(steering_action ** 2)

        # 4. Calculăm recompensa ponderată
        racing_reward = (
                self.speed_weight * speed_reward +
                self.centering_weight * lane_centering_reward +
                self.stability_weight * steering_stability
        )

        # 5. Penalizare masivă pentru coliziune
        if vehicle.crashed:
            racing_reward = -1.0

        # 6. Dacă e off-road, reward = 0
        if not vehicle.on_road:
            racing_reward = 0.0

        # Adăugăm info pentru debugging
        info['speed'] = speed
        info['speed_reward'] = speed_reward
        info['lane_centering_reward'] = lane_centering_reward
        info['racing_reward'] = racing_reward

        return obs, racing_reward, terminated, truncated, info


class EpisodeStatsCallback(BaseCallback):
    """Callback pentru colectarea statisticilor pe episoade."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
        return True


def genereaza_grafice_performanta(callback_slab, callback_bun,
                                  rezultat_slab, rezultat_bun,
                                  output_dir="plots"):
    """Genereaza 3 grafice pentru comparatie performanta."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Performanța Actor-Critic (PPO) - Comparație Model Slab vs Bun",
                 fontsize=14, fontweight='bold')

    # --- Grafic 1: Curba de invatare (reward pe episoade) ---
    ax1 = axes[0]
    if len(callback_slab.episode_rewards) > 0:
        ax1.plot(callback_slab.episode_rewards, 'r-', alpha=0.3, label='Slab (raw)')
        if len(callback_slab.episode_rewards) >= 5:
            window = min(10, len(callback_slab.episode_rewards))
            ma_slab = np.convolve(callback_slab.episode_rewards,
                                  np.ones(window) / window, mode='valid')
            ax1.plot(range(window - 1, len(callback_slab.episode_rewards)),
                     ma_slab, 'r-', linewidth=2, label='Slab (MA)')

    offset = len(callback_slab.episode_rewards)
    if len(callback_bun.episode_rewards) > 0:
        x_bun = range(offset, offset + len(callback_bun.episode_rewards))
        ax1.plot(x_bun, callback_bun.episode_rewards, 'g-', alpha=0.3, label='Bun (raw)')
        if len(callback_bun.episode_rewards) >= 5:
            window = min(10, len(callback_bun.episode_rewards))
            ma_bun = np.convolve(callback_bun.episode_rewards,
                                 np.ones(window) / window, mode='valid')
            ax1.plot(range(offset + window - 1, offset + len(callback_bun.episode_rewards)),
                     ma_bun, 'g-', linewidth=2, label='Bun (MA)')

    ax1.axvline(x=offset, color='blue', linestyle='--', alpha=0.7, label='Start Bun')
    ax1.set_xlabel('Episod')
    ax1.set_ylabel('Reward')
    ax1.set_title('Curba de Învățare')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Grafic 2: Comparatie bar chart ---
    ax2 = axes[1]
    labels = ['Slab (3k pași)', 'Bun (25k pași)']
    means = [rezultat_slab[0], rezultat_bun[0]]
    stds = [rezultat_slab[1], rezultat_bun[1]]
    colors = ['#ff6b6b', '#51cf66']

    bars = ax2.bar(labels, means, yerr=stds, capsize=10, color=colors,
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, mean, std in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 2,
                 f'{mean:.1f}±{std:.1f}', ha='center', va='bottom',
                 fontweight='bold', fontsize=10)

    ax2.set_ylabel('Reward Mediu')
    ax2.set_title('Evaluare Finală (5 episoade)')
    ax2.grid(axis='y', alpha=0.3)

    # Calculam imbunatatirea
    if rezultat_slab[0] > 0:
        improvement = ((rezultat_bun[0] - rezultat_slab[0]) / rezultat_slab[0]) * 100
        ax2.text(0.5, 0.95, f'Îmbunătățire: +{improvement:.1f}%',
                 transform=ax2.transAxes, ha='center', va='top',
                 fontsize=11, color='green', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- Grafic 3: Distributia rewardurilor ---
    ax3 = axes[2]
    data_to_plot = []
    labels_box = []

    if len(callback_slab.episode_rewards) > 0:
        data_to_plot.append(callback_slab.episode_rewards)
        labels_box.append('Slab')
    if len(callback_bun.episode_rewards) > 0:
        data_to_plot.append(callback_bun.episode_rewards)
        labels_box.append('Bun')

    if len(data_to_plot) > 0:
        bp = ax3.boxplot(data_to_plot, labels=labels_box, patch_artist=True)
        colors_box = ['#ff6b6b', '#51cf66'][:len(data_to_plot)]
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax3.set_ylabel('Reward')
    ax3.set_title('Distribuția Rewardurilor')
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "actor_critique_performance.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n>>> Grafic salvat: {output_path}")

    return output_path


def antreneaza_si_evalueaza():
    print("=" * 60)
    print("ACTOR-CRITIC (PPO) - RACING MODE 🏎️")
    print("=" * 60)

    print("\n--- 1. CONFIGURARE MEDIU ---")
    print("    Acțiuni: Steering (volan) + Accelerație/Frână")
    print("    Obiectiv: VITEZĂ MAXIMĂ!")

    # Creăm mediul cu wrapper-ul de racing
    def make_racing_env():
        env = gym.make(ENV_NAME, config=ENV_CONFIG)
        env = RacingRewardWrapper(env, speed_weight=0.6, centering_weight=0.3, stability_weight=0.1)
        return Monitor(env)

    env = make_vec_env(make_racing_env, n_envs=1)

    print("\n--- 2. ITERAȚIA 1: Antrenăm 'Grăbitul' (Actor-Critic Slab) ---")
    model_slab = ALGO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

    callback_slab = EpisodeStatsCallback()
    model_slab.learn(total_timesteps=3000, callback=callback_slab)
    model_slab.save("model_ac_slab")

    mean_reward_slab, std_reward_slab = evaluate_policy(model_slab, env, n_eval_episodes=5)
    print(f"-> Rezultat Slab: {mean_reward_slab:.2f} puncte")

    print("\n--- 3. ITERAȚIA 2: Antrenăm 'Expertul' (Actor-Critic Bun) ---")
    model_bun = ALGO("MlpPolicy", env, verbose=1, learning_rate=0.0005)

    callback_bun = EpisodeStatsCallback()
    model_bun.learn(total_timesteps=60000, callback=callback_bun)
    model_bun.save("model_ac_bun")

    mean_reward_bun, std_reward_bun = evaluate_policy(model_bun, env, n_eval_episodes=5)
    print(f"-> Rezultat Bun: {mean_reward_bun:.2f} puncte")

    # Grafic simplu (original)
    plt.figure(figsize=(10, 6))
    labels = ['Iteratia 1\n(3k Pași)', 'Iteratia 2\n(25k Pași)']
    means = [mean_reward_slab, mean_reward_bun]
    errors = [std_reward_slab, std_reward_bun]

    colors = ['red', 'green']
    bars = plt.bar(labels, means, yerr=errors, capsize=10, color=colors, alpha=0.7)

    plt.title("Performanța Actor-Critic (PPO) în funcție de Antrenament")
    plt.ylabel("Reward Mediu (Puncte)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, f"{yval:.1f}", ha='center', va='bottom',
                 fontweight='bold')

    plt.savefig("grafic_actor_critic.png")
    print("\n>>> Graficul 'grafic_actor_critic.png' a fost salvat!")
    plt.close()

    # Grafice detaliate (nou)
    genereaza_grafice_performanta(
        callback_slab, callback_bun,
        (mean_reward_slab, std_reward_slab),
        (mean_reward_bun, std_reward_bun)
    )

    env.close()
    return model_bun, callback_slab, callback_bun


def ruleaza_demo_final(n_episodes=3):
    print("\n--- 5. DEMO VIZUAL (Modelul Racer) 🏎️ ---")
    print("    Control: Steering + Accelerație/Frână")
    print("    Obiectiv: VITEZĂ MAXIMĂ!")

    # Folosim aceeași configurație ca la antrenare + wrapper racing
    env = gym.make(ENV_NAME, config=ENV_CONFIG, render_mode="human")
    env = RacingRewardWrapper(env, speed_weight=0.6, centering_weight=0.3, stability_weight=0.1)

    model = ALGO.load("model_ac_bun")

    rewards = []
    for episod in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, _ = env.step(action)
            env.render()
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episod {episod + 1} terminat. Scor: {total_reward:.1f}")

    env.close()
    return rewards


def evalueaza_si_genereaza_grafice(n_episodes=10):
    """Evalueaza modelele existente si genereaza grafice (fara antrenare)."""
    print("=" * 60)
    print("EVALUARE MODELE EXISTENTE + GENERARE GRAFICE")
    print("=" * 60)

    def make_racing_env():
        env = gym.make(ENV_NAME, config=ENV_CONFIG)
        env = RacingRewardWrapper(env, speed_weight=0.6, centering_weight=0.3, stability_weight=0.1)
        return Monitor(env)

    env = make_vec_env(make_racing_env, n_envs=1)

    # Verificam daca modelele exista
    if not os.path.exists("model_ac_slab.zip"):
        print("EROARE: model_ac_slab.zip nu exista! Ruleaza mai intai antrenarea.")
        return None
    if not os.path.exists("model_ac_bun.zip"):
        print("EROARE: model_ac_bun.zip nu exista! Ruleaza mai intai antrenarea.")
        return None

    print(f"\n--- Evaluez Model SLAB ({n_episodes} episoade) ---")
    model_slab = ALGO.load("model_ac_slab")
    rewards_slab = []
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model_slab.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            if 'episode' in info[0]:
                done = True
        rewards_slab.append(ep_reward)
        print(f"  Episod {i + 1}: {ep_reward:.1f}")

    mean_slab = np.mean(rewards_slab)
    std_slab = np.std(rewards_slab)
    print(f"-> Model SLAB: {mean_slab:.2f} ± {std_slab:.2f}")

    print(f"\n--- Evaluez Model BUN ({n_episodes} episoade) ---")
    model_bun = ALGO.load("model_ac_bun")
    rewards_bun = []
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model_bun.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            if 'episode' in info[0]:
                done = True
        rewards_bun.append(ep_reward)
        print(f"  Episod {i + 1}: {ep_reward:.1f}")

    mean_bun = np.mean(rewards_bun)
    std_bun = np.std(rewards_bun)
    print(f"-> Model BUN: {mean_bun:.2f} ± {std_bun:.2f}")

    env.close()

    # Generam graficele
    os.makedirs("plots", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Performanța Actor-Critic (PPO) - Evaluare Modele",
                 fontsize=14, fontweight='bold')

    # Grafic 1: Rewarduri pe episoade
    ax1 = axes[0]
    ax1.plot(rewards_slab, 'ro-', label='Model Slab', markersize=6, alpha=0.7)
    ax1.plot(rewards_bun, 'go-', label='Model Bun', markersize=6, alpha=0.7)
    ax1.axhline(y=mean_slab, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=mean_bun, color='green', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episod')
    ax1.set_ylabel('Reward')
    ax1.set_title('Rewarduri per Episod')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Grafic 2: Bar chart comparatie
    ax2 = axes[1]
    labels = ['Slab (3k pași)', 'Bun (25k pași)']
    means = [mean_slab, mean_bun]
    stds = [std_slab, std_bun]
    colors = ['#ff6b6b', '#51cf66']

    bars = ax2.bar(labels, means, yerr=stds, capsize=10, color=colors,
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, mean, std in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 2,
                 f'{mean:.1f}±{std:.1f}', ha='center', va='bottom',
                 fontweight='bold', fontsize=10)

    if mean_slab > 0:
        improvement = ((mean_bun - mean_slab) / abs(mean_slab)) * 100
        ax2.text(0.5, 0.95, f'Îmbunătățire: {improvement:+.1f}%',
                 transform=ax2.transAxes, ha='center', va='top',
                 fontsize=11, color='green' if improvement > 0 else 'red',
                 fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_ylabel('Reward Mediu')
    ax2.set_title(f'Evaluare ({n_episodes} episoade)')
    ax2.grid(axis='y', alpha=0.3)

    # Grafic 3: Boxplot distributie
    ax3 = axes[2]
    bp = ax3.boxplot([rewards_slab, rewards_bun], labels=['Slab', 'Bun'],
                     patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel('Reward')
    ax3.set_title('Distribuția Rewardurilor')
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = "plots/actor_critique_eval.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n>>> Grafic salvat: {output_path}")

    # Salvam si graficul original
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, means, yerr=stds, capsize=10, color=['red', 'green'], alpha=0.7)
    plt.title("Performanța Actor-Critic (PPO) în funcție de Antrenament")
    plt.ylabel("Reward Mediu (Puncte)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, f"{yval:.1f}",
                 ha='center', va='bottom', fontweight='bold')
    plt.savefig("grafic_actor_critic.png")
    plt.close()
    print(">>> Graficul 'grafic_actor_critic.png' a fost salvat!")

    print("\n" + "=" * 60)
    print("REZUMAT")
    print("=" * 60)
    print(f"Model SLAB: {mean_slab:.2f} ± {std_slab:.2f}")
    print(f"Model BUN:  {mean_bun:.2f} ± {std_bun:.2f}")
    print("=" * 60)

    return rewards_slab, rewards_bun


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Actor-Critic (PPO) pentru Racetrack")
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'demo'],
                        help='train: antrenare + grafice, eval: doar evaluare + grafice, demo: rulare vizuala')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Numar de episoade pentru evaluare')
    parser.add_argument('--no-demo', action='store_true',
                        help='Nu rula demo vizual dupa antrenare')

    args = parser.parse_args()

    if args.mode == 'train':
        antreneaza_si_evalueaza()
        if not args.no_demo:
            ruleaza_demo_final()

    elif args.mode == 'eval':
        evalueaza_si_genereaza_grafice(n_episodes=args.episodes)

    elif args.mode == 'demo':
        ruleaza_demo_final(n_episodes=args.episodes)
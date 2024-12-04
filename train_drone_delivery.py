import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from drone_delivery_env import DroneDeliveryEnv

# Créer un environnement simple (non vectorisé)
env = DroneDeliveryEnv()

# Créer un modèle PPO
model = PPO("MlpPolicy", env, verbose=1)

# Créer des listes pour stocker les valeurs d'épisode
episode_lengths = []
episode_rewards = []


# Créer une fonction callback pour enregistrer les métriques à chaque épisode
class MonitorCallback:
    def __init__(self):
        self.episode_lengths = []
        self.episode_rewards = []

    def __call__(self, locals_, globals_):
        # Récupérer la longueur et la récompense de l'épisode
        episode_len = locals_['infos'][0].get('episode', {}).get('l', 0)  # 'l' is episode length
        episode_reward = locals_['infos'][0].get('episode', {}).get('r', 0)  # 'r' is episode reward

        # Sauvegarder les valeurs dans les listes
        self.episode_lengths.append(episode_len)
        self.episode_rewards.append(episode_reward)

        return True


monitor_callback = MonitorCallback()

# Entraîner le modèle avec le callback personnalisé
total_timesteps = 200000
n_display_interval = 1000  # Afficher l'environnement tous les 1000 timesteps

for i in range(0, total_timesteps, n_display_interval):
    model.learn(total_timesteps=n_display_interval, reset_num_timesteps=False, callback=monitor_callback)

    # Afficher l'environnement après chaque intervalle d'entraînement si show_render est True
    if env.show_render:
        env.render()

# Sauvegarder le modèle
model.save("checkpoints/ppo_multi_drone_delivery_5")
print("Entraînement terminé et modèle sauvegardé !")

# Plotter l'évolution de la longueur et des récompenses des épisodes
plt.figure(figsize=(12, 6))

# Plot pour les longueurs d'épisodes
plt.subplot(1, 2, 1)
plt.plot(monitor_callback.episode_lengths)
plt.xlabel('Episode')
plt.ylabel('Episode Length')
plt.title('Evolution of Episode Length')

# Plot pour les récompenses d'épisodes
plt.subplot(1, 2, 2)
plt.plot(monitor_callback.episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Evolution of Episode Reward')

plt.tight_layout()
plt.show()

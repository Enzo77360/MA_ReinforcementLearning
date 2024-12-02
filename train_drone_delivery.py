import time
from stable_baselines3 import PPO
from drone_delivery_env import DroneDeliveryEnv

# Créer un environnement simple (non vectorisé)
env = DroneDeliveryEnv()

# Créer un modèle PPO
model = PPO("MlpPolicy", env, verbose=1)

# Entraîner le modèle et afficher l'environnement
total_timesteps = 200000
n_display_interval = 1000  # Afficher l'environnement tous les 1000 timesteps

for i in range(0, total_timesteps, n_display_interval):
    model.learn(total_timesteps=n_display_interval, reset_num_timesteps=False)

    # Afficher l'environnement après chaque intervalle d'entraînement si show_render est True
    if env.show_render:
        env.render()

# Sauvegarder le modèle
model.save("checkpoints/ppo_multi_drone_delivery_3")
print("Entraînement terminé et modèle sauvegardé !")
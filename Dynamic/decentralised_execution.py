import time
import numpy as np
from stable_baselines3 import PPO
from drone_delivery_env import DroneDeliveryEnv

# Charger l'environnement
env = DroneDeliveryEnv()
env.show_render = True  # Activer l'affichage si nécessaire
env.is_testing = False

# Charger le modèle entraîné
model_path = "checkpoints/ppo_multi_drone_delivery.zip"
model = PPO.load(model_path)

# Configurer les paramètres de test
num_episodes = 10  # Nombre d'épisodes à tester
max_steps_per_episode = 100  # Nombre maximum de pas par épisode

# Boucle de test
for episode in range(num_episodes):
    obs, _ = env.reset()  # Réinitialiser l'environnement
    total_reward = 0
    done = False
    truncated = False
    steps = 0

    print(f"--- Épisode {episode + 1} ---")
    while not (done or truncated) and steps < max_steps_per_episode:
        # Obtenir une action prédite par le modèle
        action, _states = model.predict(obs, deterministic=True)

        # Appliquer l'action et obtenir l'état suivant
        obs, reward, done, truncated, info = env.step(action)

        # Ajouter la récompense obtenue
        total_reward += reward
        steps += 1

        if env.show_render:
            env.render()

        # Pause pour ralentir l'affichage
        time.sleep(0.1)

    print(f"Épisode terminé en {steps} étapes avec une récompense totale de {total_reward:.2f}\n")

# Fermer l'environnement
env.close()
print("Tests terminés.")

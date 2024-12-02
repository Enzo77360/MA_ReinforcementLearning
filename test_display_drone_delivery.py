import numpy as np
import pygame
from stable_baselines3 import PPO
from drone_delivery_env import DroneDeliveryEnv

# Charger le modèle pré-entraîné
model = PPO.load("checkpoints/ppo_multi_drone_delivery_3")

# Créer l'environnement
env = DroneDeliveryEnv()

# Réinitialiser l'environnement et extraire l'observation
obs, _ = env.reset()  # Nous supposons ici que env.reset() renvoie un tuple (obs, info)

# Paramètre pour contrôler l'affichage pendant le test
show_render = True  # Mettre à True si vous souhaitez voir l'environnement pendant les tests

# Effectuer une série d'actions avec le modèle
total_steps = 1000  # Nombre total de steps de test
for step in range(total_steps):
    # Choisir une action basée sur l'observation courante avec le modèle
    action, _states = model.predict(obs, deterministic=True)

    # Passer l'action au modèle et recevoir la prochaine observation et la récompense
    obs, reward, done, truncated, info = env.step(action)

    # Ajouter des informations de débogage
    if done:
        print(f"Episode terminé à l'étape {step} !")
        print(f"Info: {info}")

    # Afficher l'environnement à chaque étape si show_render est True
    if show_render:
        env.render()

    # Si l'environnement est terminé, réinitialiser
    if done:
        print("Réinitialisation de l'environnement")
        obs, _ = env.reset()  # Réinitialisation de l'environnement

# Fermer proprement l'environnement
env.close()

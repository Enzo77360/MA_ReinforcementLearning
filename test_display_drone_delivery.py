import numpy as np
import pygame
from stable_baselines3 import PPO
from drone_delivery_env import DroneDeliveryEnv

# Initialisation de Pygame
pygame.init()

# Créer l'environnement
env = DroneDeliveryEnv()

# Charger le modèle préalablement entraîné
model = PPO.load("checkpoints/ppo_multi_drone_delivery_3")

# Initialiser l'écran de rendu
screen_size = 500
cell_size = screen_size // env.grid_size
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("Drone Delivery Test")

# Fonction de rendu de l'environnement
def render(env):
    screen.fill((255, 255, 255))  # Fond blanc
    for i in range(env.num_drones):
        pygame.draw.circle(screen, (0, 0, 255), (env.pos[i][0] * 50 + 25, env.pos[i][1] * 50 + 25), 10)
    for package in env.packages:
        pygame.draw.rect(screen, (0, 255, 0), (package[0] * 50 + 15, package[1] * 50 + 15, 20, 20))
    pygame.display.flip()

# Tester le modèle
obs, _ = env.reset()  # Extraire l'observation sans info

done = False
while not done:
    # Affichage de l'environnement
    render(env)

    # Exécuter une action basée sur l'observation actuelle
    action, _states = model.predict(obs, deterministic=True)

    # Appliquer l'action et obtenir la prochaine observation, la récompense et si l'épisode est terminé
    obs, reward, done, truncated, info = env.step(action)

    # Ajouter un petit délai pour rendre l'affichage visible
    pygame.time.wait(100)  # 100 ms

# Fermer l'environnement et Pygame après la simulation
pygame.quit()

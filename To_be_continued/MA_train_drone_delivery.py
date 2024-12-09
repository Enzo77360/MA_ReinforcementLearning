import numpy as np
from pettingzoo.utils.conversions import aec_to_parallel
from supersuit import pettingzoo_env_to_vec_env_v1, normalize_obs_v0
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from MA_drone_delivery_env import DroneDeliveryEnv


# Étape 1 : Configurer l'environnement
def make_env():
    """
    Crée l'environnement DroneDelivery et le prépare pour l'entraînement.
    """
    # Crée l'environnement DroneDelivery de type AEC
    env = DroneDeliveryEnv(show_pygame=False)  # Désactiver l'interface graphique pendant l'entraînement

    # Convertir un environnement AEC en environnement parallèle
    env = aec_to_parallel(env)

    # Convertir en environnement vectorisé et normaliser les observations
    env = pettingzoo_env_to_vec_env_v1(env)
    env = normalize_obs_v0(env)  # Normaliser les observations pour stabiliser l'entraînement
    return env


# Étape 2 : Créer l'environnement vectorisé
vec_env = DummyVecEnv([make_env])  # DummyVecEnv permet de gérer plusieurs instances de l'environnement

# Étape 3 : Configurer le modèle PPO
model = PPO(
    "MlpPolicy",  # Utiliser une politique avec un réseau multi-perceptron (MLP)
    vec_env,  # L'environnement vectorisé
    verbose=1,  # Afficher les informations d'entraînement dans la console
    tensorboard_log="./ppo_drone_tensorboard/",  # Dossier pour les logs TensorBoard
)

# Étape 4 : Entraîner le modèle
print("Démarrage de l'entraînement...")
model.learn(total_timesteps=100000)  # Entraînement sur 100 000 étapes
print("Entraînement terminé.")

# Étape 5 : Sauvegarder le modèle
model.save("ppo_drone_model")
print("Modèle sauvegardé sous le nom 'ppo_drone_model'.")

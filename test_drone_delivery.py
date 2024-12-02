from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from drone_delivery_env import DroneDeliveryEnv

# Charger l'environnement
env = DummyVecEnv([lambda: DroneDeliveryEnv()])  # VecEnv nécessaire pour SB3
model = PPO.load("checkpoints/ppo_multi_drone_delivery_3.zip")

# Tester sur quelques épisodes
obs = env.reset()

# Boucle sur plusieurs étapes
for episode in range(5):  # Limiter à 5 épisodes pour observer les résultats
    print(f"Début de l'épisode {episode + 1}")
    for step in range(1000):  # Limiter chaque épisode à 1000 étapes
        action, _ = model.predict(obs)  # Prédiction avec SB3
        obs, rewards, dones, infos = env.step(action)

        # Affichage des informations à chaque étape
        print(f"Étape {step + 1}:")
        print(f"Action: {action}")
        print(f"Récompense: {rewards}")
        print(f"Observation: {obs}")

        if dones[0]:  # Vérifie si l'épisode est terminé
            print(f"Épisode {episode + 1} terminé après {step + 1} étapes")
            obs = env.reset()  # Réinitialiser l'environnement
            break

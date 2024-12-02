import numpy as np
import pygame
from gymnasium import Env, spaces
import time

class DroneDeliveryEnv(Env):
    def __init__(self):
        super().__init__()
        # Paramètres de l'environnement
        self.grid_size = 10  # Taille de la grille
        self.num_drones = 2  # Nombre de drones
        self.num_packages = 3  # Nombre de colis

        # Positions initiales des drones
        self.pos = np.zeros((self.num_drones, 2), dtype=np.int32)

        # Positions des colis
        self.packages = np.random.randint(0, self.grid_size, size=(self.num_packages, 2))

        # Statut de livraison des colis
        self.delivered = np.zeros(self.num_packages, dtype=bool)

        # Espaces d'observation et d'action
        obs_dim = self.num_drones * 2 + self.num_packages * 2 + self.num_packages
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(obs_dim,), dtype=np.int32)
        self.action_space = spaces.MultiDiscrete([4] * self.num_drones)  # Une action par drone

        # Paramètres de Pygame
        self.screen_size = 500  # Taille de l'écran
        self.cell_size = self.screen_size // self.grid_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Drone Delivery")
        self.clock = pygame.time.Clock()

    def _get_obs(self):
        # Concaténer positions des drones, des colis et statut de livraison
        drone_positions = self.pos.flatten()
        package_positions = self.packages.flatten()
        delivered_status = self.delivered.astype(np.int32)
        return np.concatenate([drone_positions, package_positions, delivered_status])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Réinitialiser les positions des drones
        self.pos = np.zeros((self.num_drones, 2), dtype=np.int32)

        # Réinitialiser les positions des colis
        self.packages = np.random.randint(0, self.grid_size, size=(self.num_packages, 2))

        # Réinitialiser les statuts de livraison
        self.delivered = np.zeros(self.num_packages, dtype=bool)

        return self._get_obs(), {}

    def step(self, actions):
        rewards = np.zeros(self.num_drones)  # Récompenses pour chaque drone
        done = False
        info = {}

        for i, action in enumerate(actions):
            # Déplacement du drone
            if action == 0:  # Up
                self.pos[i][1] = max(self.pos[i][1] - 1, 0)
            elif action == 1:  # Down
                self.pos[i][1] = min(self.pos[i][1] + 1, self.grid_size - 1)
            elif action == 2:  # Left
                self.pos[i][0] = max(self.pos[i][0] - 1, 0)
            elif action == 3:  # Right
                self.pos[i][0] = min(self.pos[i][0] + 1, self.grid_size - 1)

            # Vérifier si un colis est livré
            for j, package in enumerate(self.packages):
                if not self.delivered[j] and np.array_equal(self.pos[i], package):
                    self.delivered[j] = True
                    rewards[i] += 10  # Récompense pour la livraison

        # Récompense totale pour tous les drones
        total_reward = np.sum(rewards)

        # Vérifier si toutes les livraisons sont effectuées
        if all(self.delivered):
            done = True

        # Retourner les résultats
        terminated = done  # Gymnasium sépare les cas de terminaison naturelle
        truncated = False  # Pas de troncature pour ce type d'environnement

        # Mettre à jour l'affichage Pygame
        self.render()

        return self._get_obs(), total_reward, terminated, truncated, info

    def render(self):
        # Initialiser Pygame si ce n'est pas déjà fait
        if not hasattr(self, "screen"):
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size * 50, self.grid_size * 50))
            pygame.display.set_caption("Drone Delivery")

        # Dessiner la grille et les drones
        self.screen.fill((255, 255, 255))  # Fond blanc
        for i in range(self.num_drones):
            pygame.draw.circle(self.screen, (0, 0, 255), (self.pos[i][0] * 50 + 25, self.pos[i][1] * 50 + 25), 10)

        for package in self.packages:
            pygame.draw.rect(self.screen, (0, 255, 0), (package[0] * 50 + 15, package[1] * 50 + 15, 20, 20))

        # Rafraîchir l'affichage
        pygame.display.flip()

        # Ajouter un petit délai pour éviter un rendu trop rapide
        time.sleep(0.05)

        # Boucle pour gérer la fermeture de la fenêtre
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def close(self):
        pygame.quit()


# Test d'affichage
if __name__ == "__main__":
    env = DroneDeliveryEnv()

    # Réinitialiser l'environnement
    obs = env.reset()

    # Effectuer quelques actions aléatoires et afficher l'environnement
    for _ in range(100):  # Nombre d'itérations
        # Actions aléatoires pour les drones
        actions = np.random.randint(0, 4, size=(env.num_drones,))
        obs, reward, done, truncated, info = env.step(actions)

        # Afficher l'environnement à chaque étape
        env.render()

        if done:
            break

    env.close()
    print("Test d'affichage terminé.")

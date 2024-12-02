import numpy as np
import pygame
from gymnasium import Env, spaces
import time


class DroneDeliveryEnv(Env):
    def __init__(self):
        super().__init__()

        self.grid_size = 10  # Taille de la grille (10x10)
        self.num_drones = 2  # Nombre de drones dans l'environnement
        self.num_packages = 6  # Nombre de colis à livrer
        self.pos = np.zeros((self.num_drones, 2), dtype=np.int32)  # Positions des drones, initialisées à zéro
        self.packages = np.random.randint(0, self.grid_size,
                                          size=(self.num_packages, 2))  # Positions aléatoires des colis
        self.delivered = np.zeros(self.num_packages, dtype=bool)  # Statut des colis (livré ou non), initialisé à False

        obs_dim = self.num_drones * 2 + self.num_packages * 2 + self.num_packages
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(obs_dim,), dtype=np.int32)

        self.action_space = spaces.MultiDiscrete(
            [4] * self.num_drones)  # Chaque drone peut se déplacer dans 4 directions

        # Paramètres de l'affichage Pygame
        self.screen_size = 500
        self.cell_size = self.screen_size // self.grid_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Drone Delivery")
        self.clock = pygame.time.Clock()

        self.show_render = False

    def _get_obs(self):
        """ Fonction qui génère et retourne l'observation actuelle de l'environnement. """
        drone_positions = self.pos.flatten()
        package_positions = self.packages.flatten()
        delivered_status = self.delivered.astype(np.int32)
        return np.concatenate([drone_positions, package_positions, delivered_status])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Réinitialisation des positions des drones et des colis
        self.pos = np.zeros((self.num_drones, 2), dtype=np.int32)
        self.packages = np.random.randint(0, self.grid_size, size=(self.num_packages, 2))
        self.delivered = np.zeros(self.num_packages, dtype=bool)

        return self._get_obs(), {}

    def step(self, actions):
        rewards = np.zeros(self.num_drones)
        done = False
        info = {}

        collision_penalty = -5

        move_threshold = 0.5

        # Identifier les colis non livrés
        uncollected_packages = [i for i, delivered in enumerate(self.delivered) if not delivered]

        for i, action in enumerate(actions):
            initial_pos = self.pos[i].copy()

            # Mise à jour des positions des drones en fonction de l'action
            if action == 0:  # Haut
                self.pos[i][1] = max(self.pos[i][1] - 1, 0)
            elif action == 1:  # Bas
                self.pos[i][1] = min(self.pos[i][1] + 1, self.grid_size - 1)
            elif action == 2:  # Gauche
                self.pos[i][0] = max(self.pos[i][0] - 1, 0)
            elif action == 3:  # Droite
                self.pos[i][0] = min(self.pos[i][0] + 1, self.grid_size - 1)

            # Vérification des collisions entre drones
            for j in range(self.num_drones):
                if i != j and np.array_equal(self.pos[i], self.pos[j]):
                    rewards[i] += collision_penalty
                    rewards[j] += collision_penalty

            # Vérification si un drone a collecté un colis
            for j in uncollected_packages:
                package = self.packages[j]
                if np.array_equal(self.pos[i], package):
                    self.delivered[j] = True
                    rewards[i] += 10

            # Récompense pour s'être rapproché d'un colis
            for j in uncollected_packages:
                package = self.packages[j]
                dist_before = np.linalg.norm(np.array(initial_pos) - np.array(package))
                dist_after = np.linalg.norm(np.array(self.pos[i]) - np.array(package))
                if dist_before - dist_after > move_threshold:
                    rewards[i] += 0.5

        total_reward = np.sum(rewards)

        if all(self.delivered):
            done = True

        terminated = done
        truncated = False

        if self.show_render:
            self.render()

        return self._get_obs(), total_reward, terminated, truncated, info

    def render(self):
        """ Fonction pour rendre visuellement l'état de l'environnement à l'écran. """
        self.screen.fill((255, 255, 255))

        # Dessiner les drones (en bleu)
        for i in range(self.num_drones):
            pygame.draw.circle(self.screen, (0, 0, 255), (self.pos[i][0] * 50 + 25, self.pos[i][1] * 50 + 25), 10)

        # Dessiner les colis (en vert pour ceux non livrés, en rouge pour ceux livrés)
        for j, package in enumerate(self.packages):
            if self.delivered[j]:
                pygame.draw.rect(self.screen, (255, 0, 0), (package[0] * 50 + 15, package[1] * 50 + 15, 20, 20))
            else:
                pygame.draw.rect(self.screen, (0, 255, 0), (package[0] * 50 + 15, package[1] * 50 + 15, 20, 20))

        pygame.display.update()  # Utilisez update() au lieu de flip()

        # Gérer la boucle d'événements pour fermer proprement
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.clock.tick(30)  # Limiter à 30 frames par seconde

    def close(self):
        """ Fonction pour fermer proprement Pygame. """
        pygame.quit()


# Test d'affichage
if __name__ == "__main__":
    env = DroneDeliveryEnv()

    # Réinitialiser l'environnement
    obs = env.reset()

    # Effectuer quelques actions aléatoires et afficher l'environnement
    for _ in range(1000):  # Nombre d'itérations
        # Actions aléatoires pour les drones
        actions = np.random.randint(0, 4, size=(env.num_drones,))
        obs, reward, done, truncated, info = env.step(actions)

        # Afficher l'environnement à chaque étape
        #env.render()

        if done:
            break

    env.close()
    print("Test d'affichage terminé.")

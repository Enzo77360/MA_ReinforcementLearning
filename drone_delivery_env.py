import numpy as np
import pygame
from gymnasium import Env, spaces


class DroneDeliveryEnv(Env):
    def __init__(self):
        super().__init__()

        self.grid_size = 10
        self.num_drones = 3
        self.num_packages = 6
        self.pos = np.zeros((self.num_drones, 2), dtype=np.int32)

        # Définition des colis et limites de la grille
        self.packages = np.array([[2, 3], [8, 8], [8, 2], [1, 7], [9, 5], [7, 9]])
        self.delivered = np.zeros(self.num_packages, dtype=bool)
        self.reserved_packages = [None] * self.num_drones

        self.drone_steps = np.zeros(self.num_drones, dtype=np.int32)
        obs_dim = self.num_drones * 2 + self.num_packages * 2 + self.num_packages
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(obs_dim,), dtype=np.int32)
        self.action_space = spaces.MultiDiscrete([4] * self.num_drones)

        self.screen_size = 800
        self.cell_size = self.screen_size // self.grid_size

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Drone Delivery")
        self.clock = pygame.time.Clock()

        self.show_render = False

        # Chargement des skins
        self.drone_image = pygame.image.load("images/drone.png")
        self.package_image = pygame.image.load("images/package.png")
        self.delivered_package_image = pygame.image.load("images/delivered_package.png")

        # Redimensionnement des images
        self.drone_image = pygame.transform.scale(self.drone_image, (self.cell_size, self.cell_size))
        self.package_image = pygame.transform.scale(self.package_image, (self.cell_size, self.cell_size))
        self.delivered_package_image = pygame.transform.scale(self.delivered_package_image, (self.cell_size, self.cell_size))

    def _get_obs(self):
        drone_positions = self.pos.flatten()
        package_positions = self.packages.flatten()
        delivered_status = self.delivered.astype(np.int32)
        return np.concatenate([drone_positions, package_positions, delivered_status])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Réinitialisation des colis et des drones
        self.delivered = np.zeros(self.num_packages, dtype=bool)
        self.reserved_packages = [None] * self.num_drones
        self.drone_steps = np.zeros(self.num_drones, dtype=np.int32)

        # Positionnement initial des drones avec une distance minimale
        min_distance = 2
        self.pos = np.zeros((self.num_drones, 2), dtype=np.int32)

        for i in range(self.num_drones):
            while True:
                new_pos = np.random.randint(0, self.grid_size, size=2)
                if all(np.linalg.norm(new_pos - self.pos[j]) >= min_distance for j in range(i)):
                    self.pos[i] = new_pos
                    break

        return self._get_obs(), {}

    def step(self, actions):
        rewards_personnel = np.zeros(self.num_drones)  # Récompenses personnelles
        rewards_collectives = 0  # Récompenses collectives
        done = False
        info = {}

        collision_penalty = -10
        delivery_reward = 10
        inactivity_penalty = -10
        already_delivered_penalty = -1  # Pénalité pour les drones qui passent sur un colis déjà livré
        exploration_reward = 0.1  # Petite récompense pour l'exploration d'une nouvelle case

        # Suivi des cases visitées par tous les drones
        visited_cells = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Gestion des actions des drones
        for i, action in enumerate(actions):
            self.drone_steps[i] += 1  # Augmente le compteur de déplacements

            # Appliquer le mouvement selon l'action
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
                    rewards_personnel[i] += collision_penalty
                    rewards_personnel[j] += collision_penalty

            # Vérification de la livraison d'un colis
            for package_idx, package in enumerate(self.packages):
                if not self.delivered[package_idx] and np.array_equal(self.pos[i], package):
                    self.delivered[package_idx] = True
                    rewards_personnel[i] += delivery_reward  # Récompense pour la livraison du colis
                    self.drone_steps[i] = 0  # Réinitialise le compteur après livraison

                # Pénalité si le drone passe sur un colis déjà livré
                elif self.delivered[package_idx] and np.array_equal(self.pos[i], package):
                    rewards_personnel[i] += already_delivered_penalty

            # Vérifier si la case visitée est une nouvelle case (non visitée auparavant)
            if not visited_cells[self.pos[i][0], self.pos[i][1]]:
                rewards_personnel[i] += exploration_reward  # Récompense pour l'exploration d'une nouvelle case
                visited_cells[self.pos[i][0], self.pos[i][1]] = True  # Marquer la case comme visitée

            # Appliquer la pénalité d'inactivité
            if self.drone_steps[i] > 25:
                rewards_personnel[i] += inactivity_penalty
                self.drone_steps[i] = 0  # Réinitialise le compteur après pénalité

        # Récompense collective : donner une récompense lorsque tous les colis sont livrés
        if all(self.delivered):
            rewards_collectives += 50  # Récompense collective pour avoir livré tous les colis

        # Total des récompenses pour cet épisode
        total_rewards = np.sum(rewards_personnel) + rewards_collectives

        # Vérifier si tous les colis ont été livrés
        if all(self.delivered):
            done = True

        terminated = done
        truncated = False

        if self.show_render:
            self.render()

        return self._get_obs(), total_rewards, terminated, truncated, info

    def render(self):
        # Remplir l'arrière-plan
        self.screen.fill((255, 255, 255))

        # Dessiner la grille
        for x in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.screen_size))
        for y in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_size, y))

        # Dessiner les drones
        for i in range(self.num_drones):
            self.screen.blit(
                self.drone_image,
                (self.pos[i][0] * self.cell_size, self.pos[i][1] * self.cell_size)
            )

        # Dessiner les colis
        for j, package in enumerate(self.packages):
            if not self.delivered[j]:
                self.screen.blit(
                    self.package_image,
                    (package[0] * self.cell_size, package[1] * self.cell_size)
                )
            else:
                self.screen.blit(
                    self.delivered_package_image,
                    (package[0] * self.cell_size, package[1] * self.cell_size)
                )

        # Mettre à jour l'affichage
        pygame.display.update()

        # Gérer les événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.clock.tick(30)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = DroneDeliveryEnv()
    obs = env.reset()
    for _ in range(1000):
        actions = np.random.randint(0, 4, size=(env.num_drones,))
        obs, reward, done, truncated, info = env.step(actions)
        if done:
            break
    env.close()

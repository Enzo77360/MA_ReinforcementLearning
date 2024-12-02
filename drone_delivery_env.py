import numpy as np
import pygame
from gymnasium import Env, spaces


class DroneDeliveryEnv(Env):
    def __init__(self):
        super().__init__()

        self.grid_size = 10
        self.num_drones = 2
        self.num_packages = 6
        self.pos = np.zeros((self.num_drones, 2), dtype=np.int32)

        # Définition de positions fixes pour les colis
        self.packages = np.array([[2, 3], [5, 6], [8, 2], [1, 7], [4, 5], [7, 8]])

        self.delivered = np.zeros(self.num_packages, dtype=bool)
        self.reserved_packages = [None] * self.num_drones

        self.drone_steps = np.zeros(self.num_drones, dtype=np.int32)  # Compteur des déplacements des drones

        obs_dim = self.num_drones * 2 + self.num_packages * 2 + self.num_packages
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(obs_dim,), dtype=np.int32)

        self.action_space = spaces.MultiDiscrete([4] * self.num_drones)

        self.screen_size = 500
        self.cell_size = self.screen_size // self.grid_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Drone Delivery")
        self.clock = pygame.time.Clock()

        self.show_render = False

    def _get_obs(self):
        drone_positions = self.pos.flatten()
        package_positions = self.packages.flatten()
        delivered_status = self.delivered.astype(np.int32)
        return np.concatenate([drone_positions, package_positions, delivered_status])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.zeros((self.num_drones, 2), dtype=np.int32)
        self.delivered = np.zeros(self.num_packages, dtype=bool)
        self.reserved_packages = [None] * self.num_drones
        self.drone_steps = np.zeros(self.num_drones, dtype=np.int32)  # Réinitialisation des compteurs
        return self._get_obs(), {}

    def step(self, actions):
        rewards = np.zeros(self.num_drones)
        done = False
        info = {}

        collision_penalty = -5
        move_reward = 0.1
        delivery_reward = 10
        wrong_package_penalty = -2
        inactivity_penalty = -5

        uncollected_packages = [i for i, delivered in enumerate(self.delivered) if not delivered]

        for i, action in enumerate(actions):
            self.drone_steps[i] += 1  # Augmente le compteur de déplacements

            initial_pos = self.pos[i].copy()

            if self.reserved_packages[i] is None and uncollected_packages:
                distances = [np.linalg.norm(self.pos[i] - self.packages[j]) for j in uncollected_packages]
                self.reserved_packages[i] = uncollected_packages[np.argmin(distances)]

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
                    rewards[i] += collision_penalty
                    rewards[j] += collision_penalty

            # Gestion des colis réservés
            if self.reserved_packages[i] is not None:
                package_idx = self.reserved_packages[i]
                package = self.packages[package_idx]

                if np.array_equal(self.pos[i], package):
                    if not self.delivered[package_idx]:
                        self.delivered[package_idx] = True
                        rewards[i] += delivery_reward
                        self.drone_steps[i] = 0  # Réinitialise le compteur après livraison
                    self.reserved_packages[i] = None

                else:
                    dist_before = np.linalg.norm(np.array(initial_pos) - np.array(package))
                    dist_after = np.linalg.norm(self.pos[i] - np.array(package))
                    if dist_after < dist_before:
                        rewards[i] += move_reward

            # Appliquer la pénalité d'inactivité
            if self.drone_steps[i] > 20:
                rewards[i] += inactivity_penalty
                self.drone_steps[i] = 0  # Réinitialise le compteur après pénalité

                # Réassigner une nouvelle cible au drone
                if uncollected_packages:
                    distances = [np.linalg.norm(self.pos[i] - self.packages[j]) for j in uncollected_packages]
                    self.reserved_packages[i] = uncollected_packages[np.argmin(distances)]

        # Vérifier si tous les colis ont été livrés
        if all(self.delivered):
            done = True

        terminated = done
        truncated = False

        if self.show_render:
            self.render()

        return self._get_obs(), np.sum(rewards), terminated, truncated, info

    def render(self):
        self.screen.fill((255, 255, 255))
        for i in range(self.num_drones):
            pygame.draw.circle(self.screen, (0, 0, 255), (self.pos[i][0] * 50 + 25, self.pos[i][1] * 50 + 25), 10)

        for j, package in enumerate(self.packages):
            color = (0, 255, 0) if not self.delivered[j] else (255, 0, 0)
            pygame.draw.rect(self.screen, color, (package[0] * 50 + 15, package[1] * 50 + 15, 20, 20))

        pygame.display.update()
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

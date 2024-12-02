import numpy as np
import pygame
from gymnasium import Env, spaces
import time


class DroneDeliveryEnv(Env):
    def __init__(self):
        super().__init__()
        # Paramètres de l'environnement
        self.grid_size = 10
        self.num_drones = 2
        self.num_packages = 3
        self.pos = np.zeros((self.num_drones, 2), dtype=np.int32)
        self.packages = np.random.randint(0, self.grid_size, size=(self.num_packages, 2))
        self.delivered = np.zeros(self.num_packages, dtype=bool)
        obs_dim = self.num_drones * 2 + self.num_packages * 2 + self.num_packages
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(obs_dim,), dtype=np.int32)
        self.action_space = spaces.MultiDiscrete([4] * self.num_drones)  # Une action par drone

        self.screen_size = 500
        self.cell_size = self.screen_size // self.grid_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Drone Delivery")
        self.clock = pygame.time.Clock()

        self.show_render = False  # Nouvelle variable pour contrôler l'affichage

    def _get_obs(self):
        drone_positions = self.pos.flatten()
        package_positions = self.packages.flatten()
        delivered_status = self.delivered.astype(np.int32)
        return np.concatenate([drone_positions, package_positions, delivered_status])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.zeros((self.num_drones, 2), dtype=np.int32)
        self.packages = np.random.randint(0, self.grid_size, size=(self.num_packages, 2))
        self.delivered = np.zeros(self.num_packages, dtype=bool)
        return self._get_obs(), {}

    def step(self, actions):
        rewards = np.zeros(self.num_drones)
        done = False
        info = {}

        for i, action in enumerate(actions):
            if action == 0:
                self.pos[i][1] = max(self.pos[i][1] - 1, 0)
            elif action == 1:
                self.pos[i][1] = min(self.pos[i][1] + 1, self.grid_size - 1)
            elif action == 2:
                self.pos[i][0] = max(self.pos[i][0] - 1, 0)
            elif action == 3:
                self.pos[i][0] = min(self.pos[i][0] + 1, self.grid_size - 1)

            for j, package in enumerate(self.packages):
                if not self.delivered[j] and np.array_equal(self.pos[i], package):
                    self.delivered[j] = True
                    rewards[i] += 10

        total_reward = np.sum(rewards)
        if all(self.delivered):
            done = True

        terminated = done
        truncated = False

        # Mettre à jour l'affichage Pygame si show_render est activé
        if self.show_render:
            self.render()

        return self._get_obs(), total_reward, terminated, truncated, info

    def render(self):
        if not hasattr(self, "screen"):
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size * 50, self.grid_size * 50))
            pygame.display.set_caption("Drone Delivery")

        self.screen.fill((255, 255, 255))
        for i in range(self.num_drones):
            pygame.draw.circle(self.screen, (0, 0, 255), (self.pos[i][0] * 50 + 25, self.pos[i][1] * 50 + 25), 10)

        for package in self.packages:
            pygame.draw.rect(self.screen, (0, 255, 0), (package[0] * 50 + 15, package[1] * 50 + 15, 20, 20))

        pygame.display.flip()
        time.sleep(0.05)

    def close(self):
        pygame.quit()

import numpy as np
import pygame
from pettingzoo.utils.env import AECEnv
from gymnasium import spaces

class DroneDeliveryEnv(AECEnv):
    metadata = {
        "render.modes": ["human"],
        "name": "drone_delivery_v1",
        "is_parallelizable": True,
    }

    def __init__(self, show_pygame=True, render_mode="human"):
        super().__init__()

        self.show_pygame = show_pygame
        self.render_mode = render_mode

        self.grid_size = 10
        self.num_drones = 3
        self.num_boxes = 6
        self.cell_size = 800 // self.grid_size
        self.screen_size = 800

        # Positions
        self.drone_positions = np.zeros((self.num_drones, 2), dtype=np.int32)
        self.box_positions = []
        self.collected_boxes = [False] * self.num_boxes

        self.visited_cells = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.box_counters = np.zeros(self.num_boxes, dtype=np.int32)

        self.drone_targets = [None] * self.num_drones
        self.drone_deliveries = [0] * self.num_drones
        self.drone_done = [False] * self.num_drones

        # Agents
        self.agents = [f"drone_{i}" for i in range(self.num_drones)]
        self.possible_agents = self.agents[:]
        self.rewards = {agent: 0 for agent in self.agents}

        # Spaces
        self.action_space = spaces.Discrete(4)  # Actions: Haut, Bas, Gauche, Droite
        self.observation_space = spaces.Dict({
            "drone_position": spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32),
            "box_positions": spaces.Box(low=0, high=self.grid_size - 1, shape=(self.num_boxes, 2), dtype=np.int32),
            "collected_boxes": spaces.MultiBinary(self.num_boxes),
        })

        # Graphique
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Drone Delivery")
        self.clock = pygame.time.Clock()

        # Charger et redimensionner les images
        try:
            self.drone_logo = pygame.transform.scale(
                pygame.image.load('images/drone.png'), (self.cell_size // 1.2, self.cell_size // 1.2)
            )
            self.box_logo = pygame.transform.scale(
                pygame.image.load('images/package.png'), (self.cell_size, self.cell_size)
            )
            self.collected_logo = pygame.transform.scale(
                pygame.image.load('images/delivered_package.png'), (self.cell_size, self.cell_size)
            )
        except pygame.error as e:
            print(f"Erreur de chargement des images : {e}")
            self.drone_logo = None
            self.box_logo = None
            self.collected_logo = None

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.drone_positions = np.random.randint(0, self.grid_size, size=(self.num_drones, 2))
        self.box_positions = [tuple(np.random.randint(0, self.grid_size, size=2)) for _ in range(self.num_boxes)]
        self.collected_boxes = [False] * self.num_boxes

        self._agent_selector = iter(self.agents)
        self.agent_selection = next(self._agent_selector)

        return self._observe(self.agent_selection)

    def step(self, action):
        agent_idx = int(self.agent_selection.split("_")[1])
        reward = 0

        if self.drone_done[agent_idx]:
            self.rewards[self.agent_selection] = 0
            self.agent_selection = next(self._agent_selector, None)
            if self.agent_selection is None:
                self._agent_selector = iter(self.agents)
                self.agent_selection = next(self._agent_selector)
            return

        # Déplacement du drone
        if action == 0:  # Haut
            self.drone_positions[agent_idx][1] = max(0, self.drone_positions[agent_idx][1] - 1)
        elif action == 1:  # Bas
            self.drone_positions[agent_idx][1] = min(self.grid_size - 1, self.drone_positions[agent_idx][1] + 1)
        elif action == 2:  # Gauche
            self.drone_positions[agent_idx][0] = max(0, self.drone_positions[agent_idx][0] - 1)
        elif action == 3:  # Droite
            self.drone_positions[agent_idx][0] = min(self.grid_size - 1, self.drone_positions[agent_idx][0] + 1)

        # Vérification de la collision avec un autre drone
        drone_pos = tuple(self.drone_positions[agent_idx])
        for i, other_pos in enumerate(self.drone_positions):
            if i != agent_idx and drone_pos == tuple(other_pos):
                reward -= 5
                break

        # Vérification de la livraison d'un colis
        for i, box_pos in enumerate(self.box_positions):
            if drone_pos == box_pos and not self.collected_boxes[i]:
                self.collected_boxes[i] = True
                self.box_counters[i] += 1
                reward += 10
                self.drone_targets[agent_idx] = [
                    idx for idx, collected in enumerate(self.collected_boxes) if not collected
                ]
                self.drone_deliveries[agent_idx] += 1
                if self.drone_deliveries[agent_idx] >= 2:
                    self.drone_done[agent_idx] = True
                    reward += 50
                break

        x, y = drone_pos
        if not self.visited_cells[x, y]:
            reward += 0.5
            self.visited_cells[x, y] = True

        self.rewards[self.agent_selection] = reward

        self.agent_selection = next(self._agent_selector, None)
        if self.agent_selection is None:
            self._agent_selector = iter(self.agents)
            self.agent_selection = next(self._agent_selector)

        if all(self.drone_done):
            self.reset()
            print("Tous les drones ont terminé leur mission. Simulation terminée.")

    def _observe(self, agent):
        agent_idx = int(agent.split("_")[1])
        return {
            "drone_position": self.drone_positions[agent_idx],
            "box_positions": np.array(self.box_positions),
            "collected_boxes": np.array(self.collected_boxes, dtype=np.int32),
        }

    def render(self, mode="human"):
        if not self.show_pygame:
            return

        self.screen.fill((255, 255, 255))

        for x in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.screen_size))
        for y in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_size, y))

        for pos in self.drone_positions:
            x, y = pos * self.cell_size
            if self.drone_logo:
                self.screen.blit(self.drone_logo, (x, y))
            else:
                pygame.draw.rect(self.screen, (0, 0, 255), (x, y, self.cell_size, self.cell_size))

        for i, box_pos in enumerate(self.box_positions):
            x, y = np.array(box_pos) * self.cell_size
            if self.collected_boxes[i]:
                if self.collected_logo:
                    self.screen.blit(self.collected_logo, (x, y))
                else:
                    pygame.draw.rect(self.screen, (0, 255, 0), (x, y, self.cell_size, self.cell_size))
            else:
                if self.box_logo:
                    self.screen.blit(self.box_logo, (x, y))
                else:
                    pygame.draw.rect(self.screen, (255, 0, 0), (x, y, self.cell_size, self.cell_size))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        pygame.display.update()
        self.clock.tick(30)

    def close(self):
        pygame.quit()





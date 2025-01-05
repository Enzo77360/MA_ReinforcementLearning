import numpy as np
import pygame
from gymnasium import Env, spaces

class DroneDeliveryEnv(Env):
    def __init__(self):
        super().__init__()

        self.grid_size = 10
        self.num_drones = 3
        self.water_position = np.array([0, 0])  # Fixed position of water
        self.person_position = None  # Position of the person
        self.pos = np.zeros((self.num_drones, 2), dtype=np.int32)
        self.carrying_water = [False] * self.num_drones  # Water carrying state for each drone

        self.visited_cells = np.zeros((self.grid_size, self.grid_size), dtype=bool)  # Explored cells
        self.steps_since_start = 0  # Total steps since the start
        self.steps_without_finding_person = 0  # Steps without finding the person

        # Define observation and action spaces
        obs_dim = self.num_drones * 2 + 2 + 2 + self.num_drones  # Drone positions + water + person + carrying state
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(obs_dim,), dtype=np.int32)
        self.action_space = spaces.MultiDiscrete([4] * self.num_drones)

        self.screen_size = 800
        self.cell_size = self.screen_size // self.grid_size

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Drone Delivery")
        self.clock = pygame.time.Clock()

        # Load logos
        self.drone_logo = pygame.image.load('../images/drone.png')  # Ensure the file exists
        self.person_logo = pygame.image.load('../images/package.png')
        self.person_found_logo = pygame.image.load('../images/delivered_package.png')
        self.water_logo = pygame.image.load('../images/santa.png')
        self.water_drone_logo = pygame.image.load('../images/water_drone.png')

        # Resize logos to cell size
        self.drone_logo = pygame.transform.scale(self.drone_logo, (self.cell_size // 1.2, self.cell_size // 1.2))
        self.person_logo = pygame.transform.scale(self.person_logo, (self.cell_size, self.cell_size))
        self.person_found_logo = pygame.transform.scale(self.person_found_logo, (self.cell_size, self.cell_size))
        self.water_logo = pygame.transform.scale(self.water_logo, (self.cell_size, self.cell_size))
        self.water_drone_logo = pygame.transform.scale(self.water_drone_logo, (self.cell_size // 1.2, self.cell_size // 1.2))

        self.show_render = False

        # Add the is_testing attribute to determine if it is a test
        self.is_testing = False

        self.reward_found_person = 1
        self.reward_staying_with_person = 0.01
        self.reward_deliver_water = 1
        self.reward_collective_success = 10
        self.penalty_carrying_water = -0.01  # Penalty for moving while carrying water
        self.penalty_inactivity = -0.05  # Penalty if no person is found after 20 steps
        self.reward_exploration = 0.05  # Reward for exploring a new cell

        self.person_found = False
        self.water_delivered = False
        self.water_announced = False

    def _get_obs(self):
        if self.is_testing:
            # For each drone, provide only its own position and carrying state
            drone_positions = self.pos.flatten()  # Positions of all drones
            person_position = self.person_position  # Position of the person
            water_position = np.array([0, 0])  # Position of water (empty)
            carrying_status = self.carrying_water  # Carrying state of drones

            # Add random disturbance in the observation every ~7 steps
            if np.random.rand() < 1 / 7:  # About 1 in 7 chance of introducing a disturbance
                # Disturb drone positions (set to zero)
                drone_positions = np.zeros_like(drone_positions)

            # Build observation with correct values
            observation = np.concatenate([
                drone_positions,  # Drone positions (potentially disturbed)
                water_position,  # Position of water
                person_position,  # Position of the person
                carrying_status  # Carrying state of drones
            ])

            return observation
        else:
            # During training, provide full observation (positions, water, person, carrying state)
            drone_positions = self.pos.flatten()
            carrying_status = np.array(self.carrying_water, dtype=np.int32)
            return np.concatenate([drone_positions, self.water_position, self.person_position, carrying_status])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate a random position for the person
        while True:
            person_position = np.random.randint(0, self.grid_size, size=2)
            if not np.array_equal(person_position, self.water_position):
                break
        self.person_position = person_position

        self.person_found = False
        self.water_delivered = False
        self.carrying_water = [False] * self.num_drones
        self.water_announced = False

        self.pos = np.random.randint(0, self.grid_size, size=(self.num_drones, 2))
        self.visited_cells = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.steps_since_start = 0
        self.steps_without_finding_person = 0

        return self._get_obs(), {}

    def step(self, actions):
        rewards = np.zeros(self.num_drones)
        done = False
        info = {}

        self.steps_since_start += 1
        self.steps_without_finding_person += 1

        for i, action in enumerate(actions):
            if action == 0:  # Up
                self.pos[i][1] = max(self.pos[i][1] - 1, 0)
            elif action == 1:  # Down
                self.pos[i][1] = min(self.pos[i][1] + 1, self.grid_size - 1)
            elif action == 2:  # Left
                self.pos[i][0] = max(self.pos[i][0] - 1, 0)
            elif action == 3:  # Right
                self.pos[i][0] = min(self.pos[i][0] + 1, self.grid_size - 1)

            if self.carrying_water[i]:
                rewards[i] += self.penalty_carrying_water

            if not self.visited_cells[self.pos[i][0], self.pos[i][1]]:
                rewards[i] += self.reward_exploration
                self.visited_cells[self.pos[i][0], self.pos[i][1]] = True

        for i in range(self.num_drones):
            if not self.person_found and np.array_equal(self.pos[i], self.person_position):
                self.person_found = True
                rewards[i] += self.reward_found_person
                self.steps_without_finding_person = 0
                info['person_position_shared'] = self.person_position

        if self.person_found and not self.water_delivered:
            for i in range(self.num_drones):
                if np.array_equal(self.pos[i], self.person_position):
                    rewards[i] += self.reward_staying_with_person

        # Manage water pickup
        for i in range(self.num_drones):
            if not self.carrying_water[i] and np.array_equal(self.pos[i], self.water_position):
                if not self.water_announced:  # First drone to pick up water
                    if not self.person_found:
                        rewards[i] -= 0.2
                    else:
                        rewards[i] += 2
                    self.carrying_water[i] = True
                    self.water_announced = True  # Announce that water was picked up
                else:  # Water already picked up, apply penalty
                    rewards[i] -= 0.01

        for i in range(self.num_drones):
            if self.carrying_water[i] and np.array_equal(self.pos[i], self.person_position):
                self.water_delivered = True
                self.water_available = False
                rewards += self.reward_deliver_water / self.num_drones
                self.carrying_water[i] = False

        if self.person_found and self.water_delivered:
            rewards += self.reward_collective_success / self.num_drones
            done = True

        if self.steps_without_finding_person >= 20 and not self.person_found:
            rewards += self.penalty_inactivity

        if self.show_render:
            self.render()

        return self._get_obs(), np.sum(rewards), done, False, info

    def render(self):
        self.screen.fill((255, 255, 255))

        # Draw the grid
        for x in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.screen_size))
        for y in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_size, y))

        # Center drones
        for i in range(self.num_drones):
            # Calculate offset to center the image on the cell
            drone_x = self.pos[i][0] * self.cell_size + (self.cell_size - self.drone_logo.get_width()) // 2
            drone_y = self.pos[i][1] * self.cell_size + (self.cell_size - self.drone_logo.get_height()) // 2
            if self.carrying_water[i]:
                self.screen.blit(self.water_drone_logo, (drone_x, drone_y))
            else:
                self.screen.blit(self.drone_logo, (drone_x, drone_y))

        # Center the person
        person_x = self.person_position[0] * self.cell_size + (self.cell_size - self.person_logo.get_width()) // 2
        person_y = self.person_position[1] * self.cell_size + (self.cell_size - self.person_logo.get_height()) // 2
        if self.person_found:
            self.screen.blit(self.person_found_logo, (person_x, person_y))
        else:
            self.screen.blit(self.person_logo, (person_x, person_y))

        # Center the water
        water_x = self.water_position[0] * self.cell_size + (self.cell_size - self.water_logo.get_width()) // 2
        water_y = self.water_position[1] * self.cell_size + (self.cell_size - self.water_logo.get_height()) // 2
        self.screen.blit(self.water_logo, (water_x, water_y))

        pygame.display.update()

        # Handle events
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

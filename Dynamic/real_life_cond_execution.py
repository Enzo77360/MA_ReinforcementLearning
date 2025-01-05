import time
import numpy as np
from stable_baselines3 import PPO
from drone_delivery_env import DroneDeliveryEnv

# Load the environment
env = DroneDeliveryEnv()
env.show_render = True  # Enable rendering if needed
env.is_testing = False  # Set the environment to testing mode

# Load the trained model
model_path = "checkpoints/ppo_multi_drone_delivery.zip"
model = PPO.load(model_path)

# Configure test parameters
num_episodes = 10  # Number of episodes to test
max_steps_per_episode = 100  # Maximum number of steps per episode

# Testing loop
for episode in range(num_episodes):
    obs, _ = env.reset()  # Reset the environment
    total_reward = 0
    done = False
    truncated = False
    steps = 0

    print(f"--- Episode {episode + 1} ---")
    while not (done or truncated) and steps < max_steps_per_episode:
        # Get an action predicted by the model
        action, _states = model.predict(obs, deterministic=True)

        # Apply the action and get the next state
        obs, reward, done, truncated, info = env.step(action)

        # Accumulate the reward obtained
        total_reward += reward
        steps += 1

        if env.show_render:
            env.render()

        # Pause to slow down the rendering
        time.sleep(0.1)

    print(f"Episode completed in {steps} steps with a total reward of {total_reward:.2f}\n")

# Close the environment
env.close()
print("Testing completed.")

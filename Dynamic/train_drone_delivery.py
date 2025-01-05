import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from drone_delivery_env import DroneDeliveryEnv  # Ensure DroneDeliveryEnv is in the same directory or accessible

# Create the DroneDeliveryEnv environment
env = DroneDeliveryEnv()

# Define whether the environment should display an animation (render) during training
env.show_render = False  # Set to False if you do not want to display the render

# Create a PPO model (Proximal Policy Optimization)
model = PPO("MlpPolicy", env, verbose=1)  # Uses a policy based on Multi-Layer Perceptrons (MLPs)

# Training parameters
total_timesteps = 100000  # Total number of training steps
n_display_interval = 1000  # Interval to display the render

# Training loop
for i in range(0, total_timesteps, n_display_interval):
    # Train the model in intervals
    model.learn(total_timesteps=n_display_interval, reset_num_timesteps=False)

    # Display the environment after each training interval if show_render is enabled
    if env.show_render:
        env.render()

# Save the trained model
model.save("checkpoints/ppo_multi_drone_delivery_test")
print("Training completed and model saved!")

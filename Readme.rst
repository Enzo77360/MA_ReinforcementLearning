# Multi-Agent Reinforcement Learning Project

## Overview
This project implements a Multi-Agent Reinforcement Learning (MARL) environment where multiple drones collaborate to locate a person in need and deliver water in a grid-based simulation. The project leverages **Stable-Baselines3** for training the agents using Proximal Policy Optimization (PPO).

The environment includes:
- A grid-based map representing the world.
- Multiple drones acting as agents.
- A person positioned randomly within the grid.
- A water source fixed at a predefined position.

Drones must coordinate to locate the person, deliver water, and maximize rewards through exploration, teamwork, and task completion.

---

## Environment Setup

### Dependencies
Ensure the following libraries are installed:
- **Python 3.8+**
- **NumPy**
- **PyGame**
- **Matplotlib**
- **Stable-Baselines3**
- **Gymnasium**

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Enzo77360/MA_ReinforcementLearning.git

   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that the `images` directory contains the required images for rendering, including:
   - `drone.png`
   - `package.png`
   - `delivered_package.png`
   - `santa.png`
   - `water_drone.png`

4. Run the code in a Python environment (e.g., PyCharm, Jupyter Notebook).

---

## Project Components

### 1. Environment Design

The custom environment (`DroneDeliveryEnv`) is implemented in `drone_delivery_env.py`. It follows the **OpenAI Gymnasium API** structure and includes the following features:

#### Grid Setup
- **Grid size:** The environment is represented as a 10x10 grid.
- **Drones:** 3 drones, each with its own position and state.
- **Water source:** Fixed at `(0, 0)`.
- **Person:** Randomly placed at the start of each episode.

#### Observations
- Includes drone positions, water source position, person position, and carrying status of each drone.

#### Actions
Each drone can:
- Move in four directions: Up, Down, Left, Right.

#### Rewards
- **Finding the person:** Positive reward.
- **Delivering water:** Positive reward.
- **Exploration:** Positive reward for visiting new cells.
- **Carrying water penalty:** Negative reward for excessive movement while carrying water.
- **Inactivity penalty:** Negative reward for delays in finding the person.

#### Rendering
PyGame is used to render the environment, displaying the grid, drone positions, the person, and the water source.

### 2. Training the Model

Training is performed using **Stable-Baselines3 PPO**. The script is located in `train.py` and includes the following steps:

#### Steps
1. **Initialize the environment:**
   ```python
   from drone_delivery_env import DroneDeliveryEnv
   env = DroneDeliveryEnv()
   ```

2. **Set up PPO model:**
   ```python
   from stable_baselines3 import PPO
   model = PPO("MlpPolicy", env, verbose=1)
   ```

3. **Train the model:**
   ```python
   model.learn(total_timesteps=100000)
   ```

4. **Save the model:**
   ```python
   model.save("checkpoints/ppo_multi_drone_delivery")
   ```

#### Training Parameters
- **Total timesteps:** 100,000.
- **Rendering interval:** The environment can optionally render during training for visual feedback.

#### Running the Script
To train the model, execute:
```bash
python train.py
```

---

### 3. Testing and Visualization

Testing the trained model is performed using the script `test.py`, which evaluates the model in a testing mode.

#### Testing Steps
1. **Load the trained model:**
   ```python
   from stable_baselines3 import PPO
   model = PPO.load("checkpoints/ppo_multi_drone_delivery.zip")
   ```

2. **Run test episodes:**
   ```python
   for episode in range(num_episodes):
       obs = env.reset()
       while not done:
           action, _states = model.predict(obs, deterministic=True)
           obs, reward, done, info = env.step(action)
           env.render()
   ```

3. **Close the environment:**
   ```python
   env.close()
   ```

#### Visualization
- PyGame renders the environment, showing drone movements, water delivery, and task completion.
- The rendering is slowed down using `time.sleep()` to observe the drones' behavior step-by-step.

#### Running the Script
To test the model, execute:
```bash
python test.py
```

---

## Project Structure

```
multi-agent-rl/
|-- checkpoints/            # Saved models
|-- images/                 # Icons for rendering
|-- drone_delivery_env.py   # Environment implementation
|-- train.py                # Training script
|-- test.py                 # Testing script
|-- README.md               # Project documentation
|-- requirements.txt        # Dependencies
```

---

## Future Improvements
- **Dynamic environment:** Introduce dynamic obstacles or moving targets.
- **Communication:** Enable drones to share observations for better coordination.
- **Scalability:** Extend to larger grids or more agents.
- **Reward tuning:** Experiment with different reward functions for optimized performance.

---

## Conclusion
This project demonstrates the application of Multi-Agent Reinforcement Learning to a collaborative task. The drones learn to locate and assist a person efficiently through exploration and teamwork. The framework can be extended to more complex real-world scenarios.


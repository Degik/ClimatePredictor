# NumPy
import numpy as np
# Gymnasium is a custom library for creating custom environments
import gymnasium as gym
from gymnasium import spaces

class ClimateEnv(gym.Env):
    def __init__(self, data):
        """
        Initialize the environment with the given dataset.
        """
        super(ClimateEnv, self).__init__()

        self.data = data
        self.current_step = 0
        
        obs_dim = len(self.data.columns) - 2  # Remove target columns
        self.observation_space = spaces.Box(
            low=-9999, high=9999, shape=(obs_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(2)  # 0: temperature, 1: precipitation

    def reset(self, seed=None, options=None):
        """
        Reset the environment and return the initial observation.
        """
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.data.iloc[self.current_step][:-2].values.astype(np.float32)
        return obs, {}  # Gymnasium requires an info dict

    def step(self, action):
        """
        Execute one step in the environment based on the chosen action.
        """
        if action == 0:
            # Prediction for temperature
            reward = self._calculate_reward_temp()
        elif action == 1:
            # Prediction for precipitation
            reward = self._calculate_reward_precip(method='log_loss')
        else:
            raise ValueError("Invalid action!")

        # Move to the next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        truncated = False  # No truncation condition

        if not done:
            obs = self.data.iloc[self.current_step][:-2].values.astype(np.float32)
        else:
            obs, _ = self.reset()  # Reset the environment when done
        
        return obs, reward, done, truncated, {}  # Gymnasium requires 5 return values

    def _calculate_reward_temp(self):
        """
        Reward calculation for temperature prediction.
        """
        true_temp = self.data.iloc[self.current_step]['Target_Temperature']
        predicted_temp = np.random.uniform(-10, 50)  # Placeholder for model prediction
        reward = -np.abs(predicted_temp - true_temp)  # Negative absolute error as reward
        return reward

    def _calculate_reward_precip(self, method='binary'):
        """
        Reward calculation for precipitation prediction.
        """
        true_precip = self.data.iloc[self.current_step]['Rain_Tomorrow']
        predicted_precip = np.random.uniform(0, 1)  # Placeholder for model prediction

        if method == 'binary':
            # Reward is 1 if correct, -1 otherwise
            reward = 1 if (predicted_precip > 0.5) == true_precip else -1
        elif method == 'log_loss':
            # Logarithmic loss (penalizes incorrect predictions more heavily)
            reward = -np.log(predicted_precip) if true_precip == 1 else -np.log(1 - predicted_precip)
        
        return reward
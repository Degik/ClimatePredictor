import gym
import numpy as np
from gym import spaces

class ClimateEnv(gym.Env):
    def __init__(self, data):
        super(ClimateEnv, self).__init__()

        self.data = data
        self.current_step = 0
        
        obs_dim = len(self.data.columns) - 2 # Remove the target columns
        self.observation_space = spaces.Box(
            low=-9999, high=9999, shape=(obs_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(2) # 0: temperature, 1: precipitation

    def reset(self):
        """ Reset the environment """
        self.current_step = 0
        return self.data.iloc[self.current_step][:-2].values

    def step(self, action):
        """ Execute one step within the environment """
        
        if action == 0:
            # Prediction for temperature
            reward = self._calculate_reward_temp()
        elif action == 1:
            # Prediction for precipitation
            reward = self._calculate_reward_precip(method='log_loss')
        else:
            raise ValueError("Invalid action!")

        # Take the next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        if not done:
            obs = self.data.iloc[self.current_step][:-2].values
        else:
            obs = self.reset()
        
        return obs, reward, done, {}

    def _calculate_reward_temp(self):
        """ Reward for the temperature prediction """
        true_temp = self.data.iloc[self.current_step]['Target_Temperature']
        predicted_temp = np.random.uniform(-10, 50)
        reward = -np.abs(predicted_temp - true_temp)
        return reward

    def _calculate_reward_precip(self, method='binary'):
        """ Reward for the precipitation prediction """
        true_precip = self.data.iloc[self.current_step]['Rain_Tomorrow']
        predicted_precip = np.random.uniform(0, 1)

        if method == 'binary':
            # Reward 1 if the prediction is correct, -1 otherwise
            reward = 1 if (predicted_precip > 0.5) == true_precip else -1
        elif method == 'log_loss':
            # Logarithmic loss
            # https://scikit-learn.org/stable/modules/model_evaluation.html#log-loss
            reward = -np.log(predicted_precip) if true_precip == 1 else -np.log(1 - predicted_precip)
        return reward
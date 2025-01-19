import gym
import numpy as np

class ClimateEnv(gym.Env):
    def __init__(self, data):
        self.data = data
        self.observation_space = np.array([1])
        self.action_space = np.array([1])
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step][:-2].values

    def step(self, action):
        reward = self._calculate_reward(action)
        done = self.current_step == len(self.data) - 1
        self.current_step += 1
        if not done:
            obs = self.data.iloc[self.current_step][:-2].values
        else:
            obs = None
        return obs, reward, done, {}

    def _calculate_reward(self, action):
        predicted_temp = self.model.predict(self.state)
        true_temp = self.data.iloc[self.current_step]['Target_Temperature']
        reward = -np.abs(predicted_temp - true_temp)
        return reward
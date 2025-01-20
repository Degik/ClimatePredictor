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
        # Select the action
        if action == 0:  
            # Temperature prediction action
            reward = self._calculate_reward_temp(action)
        elif action == 1: 
            # Precipitation prediction action
            reward = self._calculate_reward_precip(action, method='log_loss') # method='binary' or 'log_loss'
        else:
            raise ValueError("Invalid action!")

        # Update the state
        self.current_step += 1
        done = self.current_step == len(self.data) - 1
        
        if not done:
            obs = self.data.iloc[self.current_step][:-2].values
        else:
            obs = None

        # Restituisci i valori
        return obs, reward, done, {}

    def _calculate_reward_temp(self, action):
        predicted_temp = self.model.predict(self.state)
        true_temp = self.data.iloc[self.current_step]['Target_Temperature']
        reward = -np.abs(predicted_temp - true_temp)
        return reward
    
    def _calculate_reward_precip(self, action, method = 'binary'):
        predicted_precip = self.model.predict(self.state)
        true_precip = self.data.iloc[self.current_step]['Rain_Tomorrow']
        if method == 'binary':
            # Reward is 1 if the prediction is correct, -1 otherwise
            reward = 1 if predicted_precip == true_precip else -1
        if method == 'log_loss':
            # Reward is -log(p) if true_precip == 1, -log(1 - p) otherwise
            reward = -np.log(predicted_precip) if true_precip == 1 else -np.log(1 - predicted_precip)
        return reward
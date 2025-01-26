# NumPy
import numpy as np
# Gymnasium is a custom library for creating custom environments
import gymnasium as gym
from gymnasium import spaces
# Torch
import torch
import torch.nn.functional as F

class ClimateEnv(gym.Env):
    def __init__(self, data):
        """
        Initialize the environment with the given dataset.
        
        Args:
            data (pd.DataFrame): The local climate dataset with columns:
                [
                    "HourlyVisibility",
                    "HourlyStationPressure",
                    "HourlyRelativeHumidity",
                    "HourlyWindDirection",
                    "HourlyWindSpeed",
                    "HourlyAltimeterSetting",
                    "HourlyWetBulbTemperature",
                    "HourlyDewPointTemperature",
                    "HourlyDryBulbTemperature",
                    "DATE"
                ]
        """
        super(ClimateEnv, self).__init__()

        self.data = data
        self.current_step = 0

        # Identify feature columns (exclude the target and DATE)
        self.feature_columns = [
            "HourlyVisibility",
            "HourlyStationPressure",
            "HourlyRelativeHumidity",
            "HourlyWindDirection",
            "HourlyWindSpeed",
            "HourlyAltimeterSetting",
            "HourlyWetBulbTemperature",
            "HourlyDewPointTemperature"
        ]
        self.target_column = "HourlyDryBulbTemperature"
        
        # Observation space: 8 continuous features
        self.observation_space = spaces.Box(
            low=-9999.0,
            high=9999.0,
            shape=(len(self.feature_columns),),
            dtype=np.float32
        )

        # Action space: continuous prediction of temperature (for example, -50°C to 60°C)
        self.action_space = spaces.Box(
            low=np.array([-50.0]),
            high=np.array([95.0]),
            shape=(1,),
            dtype=np.float32
        )

    def update_end_date(self, new_date):
        """
        Adjust the subset of self.data that the environment uses for rollouts.
        This avoids re-creating or re-registering the environment.
        """
        if self.data.empty:
            self.current_data = self.data
        else:
            self.current_data = self.data[self.data["DATE"] <= new_date].copy()
        self.current_step = 0

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial step and return the first observation.
        
        Returns:
            obs (np.ndarray): The feature vector at the current step.
            info (dict): Additional info dictionary (empty in this case).
        """
        super().reset(seed=seed)
        self.current_step = 0

        # Build the first observation
        obs = self._get_observation(self.current_step)
        return obs, {}

    
    def step(self, action):
        """
        Execute one step in the environment based on the chosen action.

        Args:
            action (np.ndarray): The predicted temperature (shape = (1,)).

        Returns:
            obs (np.ndarray): Next observation.
            reward (float): Reward for the action taken.
            done (bool): True if the episode is over.
            truncated (bool): True if the episode was truncated.
            info (dict): Additional debugging info.
        """
        # Convert action to PyTorch tensor
        action_tensor = torch.tensor(action, dtype=torch.float32)
        predicted_temp = action_tensor.item() if action_tensor.numel() > 0 else 0.0
        predicted_temp = torch.tensor(predicted_temp).clamp(min=-20, max=95).item() # Block predictions outside the range

        # Get the true temperature for the current step
        true_temp = self.data[self.target_column].iloc[self.current_step]

        # Check for NaN values in the true temperature
        if np.isnan(true_temp):
            print(f"[WARNING] NaN detected at step {self.current_step}, assigning penalty")
            reward = -10  # Penalization for missing data
        else:
            true_temp_tensor = torch.tensor(true_temp, dtype=torch.float32).unsqueeze(0)

            # Mean Squared Error (MSE) Loss
            mse_loss = F.mse_loss(action_tensor, true_temp_tensor)

            # Mean Absolute Error (MAE) Loss
            mae_loss = F.l1_loss(action_tensor, true_temp_tensor)

            # Stability Penalty: L1 loss between current and previous prediction
            if self.current_step > 0:
                prev_pred_tensor = torch.tensor(self.prev_action, dtype=torch.float32)
                stability_penalty = F.l1_loss(action_tensor.view(-1), prev_pred_tensor.view(-1)) / 10
            else:
                stability_penalty = torch.tensor(0.0)

            # Reward is the negative sum of losses
            reward = -(mse_loss + mae_loss + stability_penalty).item()

        # Store the current action for the next step
        self.prev_action = predicted_temp

        # Move to the next step
        self.current_step += 1
        done = self.current_step >= (len(self.data) - 1)
        truncated = False

        # Reset the environment if the episode is over
        if done:
            obs, _ = self.reset()
        else:
            obs = self._get_observation(self.current_step)


        if self.current_step % 1000 == 0:
            print(f"[DEBUG] Step {self.current_step} | True Temp: {true_temp:.2f} | Pred: {predicted_temp:.2f} | Reward: {reward:.4f}")

        return obs, reward, done, truncated, {}

    def _get_observation(self, step_idx):
        """
        Return the feature vector at the given step index.
        
        Args:
            step_idx (int): The current index in the dataset.
        
        Returns:
            np.ndarray: The selected feature values as a float32 array.
        """
        row = self.data.loc[step_idx, self.feature_columns]
        return row.values.astype(np.float32)
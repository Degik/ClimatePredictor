# NumPy
import numpy as np
# Gymnasium is a custom library for creating custom environments
import gymnasium as gym
from gymnasium import spaces

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
            """
            # Extract the true temperature
            true_temp = self.data[self.target_column].iloc[self.current_step]
            # The agent's predicted temperature
            predicted_temp = float(action[0])

            # Compute reward
            reward = -abs(predicted_temp - true_temp)

            # Move to the next step
            self.current_step += 1
            done = self.current_step >= (len(self.data) - 1)

            if not done:
                obs = self._get_observation(self.current_step)
            else:
                # If done, reset environment automatically
                obs, _ = self.reset()

            return obs, reward, done, {}

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
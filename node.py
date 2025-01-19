import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
import numpy as np

# Environment
from gym import spaces
from ClimateEnvironment import ClimateEnv

class Node:
    def __init__(self, config):
        # Initialize the environment and the node configuration
        self.config = config
        self.env = ClimateEnv(config)
        self.agents = self.env.agents
        # Initialize the model
        print("Node initialized with config: ", config)
        # Initialize the model with the environment
        # Two agents: temp_agent and precip_agent
        self.agents = ["temp_agent", "precip_agent"]
        # Space observation: Box(-9999, 9999, (n_features,))
        obs_dim = len(self.env.observation_cols)
        self.obs_space = spaces.Box(
            low=-9999, high=9999,
            shape=(obs_dim,),
            dtype=np.float32
        )
        # Policy config using PPO, we need to predict the next day temperature and precipitation probability
        self.policy_config = PPOConfig(
            num_workers=1,
            num_envs_per_worker=1,
            rollouts_used=1,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh"
            },
            framework="torch"
        )
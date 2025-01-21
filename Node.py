import os
import pandas as pd
# RLlib
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
# Environment
from ClimateEnvironment import ClimateEnv

@ray.remote
class Node:
    def __init__(self, node_id, local_data_path):
        """
        Each node is responsible for training a PPO agent on a local dataset.
        """
        self.node_id = node_id
        self.data = None
        try :
            self.data = pd.read_csv(local_data_path)  # Load local data
        except:
            print('File not found in the path:', local_data_path)
            
        self.env = ClimateEnv(self.data)
        # Register the environment
        #register_env("ClimateEnv", lambda config: ClimateEnv(self.data))

        # PPO policy configuration
        self.config = (
            PPOConfig()
            .framework("torch")
            .training(gamma=0.99, lr=0.0003, train_batch_size=4000)
            .env_runners(num_env_runners=1, env_runner_cls=ClimateEnv)
            .resources(num_gpus=0)  # CPUs-based
        )

        # Trainer PPO
        self.trainer = self.config.build(env=self.env)
        print(f"Node {self.node_id} initialized.")

    def train(self, num_steps=1):
        """
        Train the PPO agent for a given number of steps.
        """
        for _ in range(num_steps):
            result = self.trainer.train()
            print(f"Node {self.node_id} - Reward mean: {result['episode_reward_mean']}")

    def get_weights(self):
        """
        Return the weights of the local model.
        This method is used by the FederatedAggregator to aggregate the weights.
        """
        return self.trainer.get_policy().get_weights()

    def set_weights(self, global_weights):
        """
        Set the weights of the local model.
        This method is used by the FederatedAggregator to update the local weights.
        """
        self.trainer.get_policy().set_weights(global_weights)
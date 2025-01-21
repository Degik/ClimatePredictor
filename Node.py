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
        Each node trains a PPO agent on its local dataset.
        """
        self.node_id = node_id
        self.data = None
        try:
            self.data = pd.read_csv(local_data_path)  # Load local data
        except FileNotFoundError:
            print(f"File not found at path: {local_data_path}")

        # Register the environment dynamically on each worker
        register_env("ClimateEnv", lambda config: ClimateEnv(self.data))

        # PPO policy configuration
        self.config = (
            PPOConfig()
            .framework("torch")
            .training(gamma=0.99, lr=0.0003, train_batch_size=4000)
            .env("ClimateEnv")  # Use the registered environment name
            .resources(num_gpus=0)  # CPU-based
        )

        # Trainer PPO
        self.trainer = self.config.build()
        print(f"✅ Node {self.node_id} initialized.")

    def train(self, num_steps=1):
        """
        Train the PPO agent for a given number of steps.
        """
        for _ in range(num_steps):
            result = self.trainer.train()
            print(f"Node {self.node_id} - Reward mean: {result['episode_reward_mean']}")

    def get_weights(self):
        """
        Return the weights of the local model for federated learning.
        """
        return self.trainer.get_policy().get_weights()

    def set_weights(self, global_weights):
        """
        Update the local model with the federated weights.
        """
        self.trainer.get_policy().set_weights(global_weights)
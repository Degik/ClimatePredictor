import os
import collections

import ray
import pandas as pd

# RLlib
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

# Environment
from ClimateEnvironment import ClimateEnv

@ray.remote
class Node:
    def __init__(self, node_id, local_data_path, start_date):
        """
        Each node trains a PPO agent on its local dataset.
        
        Args:
            node_id (int): Unique identifier for this node (station).
            local_data_path (str): Path to the CSV file containing the local dataset.
            start_date (pd.Timestamp or str): The initial cutoff date for the training data.

        Returns:
            None
        """
        self.node_id = node_id
        self.local_data_path = local_data_path

        # Load the full dataset
        try:
            self.full_data = pd.read_csv(local_data_path)
        except FileNotFoundError:
            print(f"File not found at path: {local_data_path}")
            self.full_data = pd.DataFrame()

        # Convert DATE column to datetime for date-based filtering
        if not self.full_data.empty:
            self.full_data["DATE"] = pd.to_datetime(self.full_data["DATE"])
        else:
            print(f"Node {self.node_id} has empty dataset!")

        # Define the initial "end date" for the training data
        self.current_end_date = None
        if not self.full_data.empty:
            self.current_end_date = pd.to_datetime(start_date)
            if self.current_end_date < self.full_data["DATE"].min():
                # If start_date is before the min date in the dataset,
                # adjust it to the min date to avoid empty subsets.
                self.current_end_date = self.full_data["DATE"].min()

        # Create a unique environment name for this node
        env_name = f"ClimateEnv_{self.node_id}"


        # We neeed create a new ClimateEnv with the full dataset,
        # and we will call `update_end_date` later to limit it.
        def env_creator(cfg):
            env = ClimateEnv(self.full_data)
            env.update_end_date(self.current_end_date)
            return env

        # Register the environment ONCE
        register_env(env_name, env_creator)

        # PPO configuration
        self.config = (
            PPOConfig()
            .framework("torch")
            .environment(env_name)
            .resources(num_gpus=0)
            .training(gamma=0.99, lr=0.0003, train_batch_size=4000)
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False
            )
        )

        # Build the trainer
        self.trainer = self.config.build()

        print(f"[Node {self.node_id}] Initialized with end date: {self.current_end_date}.")

    def add_new_days(self, days=1):
        """
        Extend the current_end_date by the specified number of days and update the environment.
        """
        if self.full_data.empty:
            print(f"[Node {self.node_id}] No data available to reveal.")
            return

        max_date = self.full_data["DATE"].max()
        self.current_end_date = min(self.current_end_date + pd.Timedelta(days=days), max_date)

        print(f"[Node {self.node_id}] Updating end date to: {self.current_end_date}")

        try:
            print(f"[Node {self.node_id}] Workers: {self.trainer.workers()}")
        except Exception as e:
            print(f"[Node {self.node_id}] ERROR accessing workers: {e}")

        new_end_date = self.current_end_date

        def do_update(env):
            try:
                env.update_end_date(new_end_date)
            except Exception as e:
                print(f"[Node {self.node_id}] ERROR updating environment: {e}")

        try:
            for worker in self.trainer.workers():
                worker.foreach_env(do_update)
            print(f"[Node {self.node_id}] Successfully updated worker environments.")
        except Exception as e:
            print(f"[Node {self.node_id}] ERROR updating workers: {e}")

    def train(self, num_steps=1):
        """
        Train the PPO agent for a specified number of steps (iterations).
        
        Args:
            num_steps (int): Number of training iterations to run.
        
        Returns:
            dict: Training results from RLlib.
        """
        for i in range(num_steps):
            result = self.trainer.train()
            print(f"[Node {self.node_id}] Iteration {i}: {result['episode_reward_mean']}")

    def get_weights(self):
        """
        Return the weights of the local model for federated learning.
        
        Returns:
            OrderedDict: The local model weights, in the same order for each node.
        """
        weights = self.trainer.get_weights()
        weights = collections.OrderedDict(weights)
        return weights

    def set_weights(self, global_weights):
        """
        Update the local model with the federated global weights.
        
        Args:
            global_weights (dict): A dictionary of weights from the global model.

        Returns:
            None
        """
        self.trainer.set_weights(global_weights)
        print(f"[Node {self.node_id}] Weights updated with global model.")
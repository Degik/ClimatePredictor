import os
import pandas as pd
# RLlib
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
# Environment
from ClimateEnvironment import ClimateEnv
# Utilities
import collections

@ray.remote
class Node:
    def __init__(self, node_id, local_data_path, start_date):
        """
        Each node trains a PPO agent on its local dataset.
        """
        self.node_id = node_id
        try:
            self.full_data = pd.read_csv(local_data_path)
        except FileNotFoundError:
            print(f"File not found at path: {local_data_path}")
            self.full_data = pd.DataFrame()

        # Convert 'DATE' to datetime (needed for day time increment)
        if not self.full_data.empty:
            self.full_data["DATE"] = pd.to_datetime(self.full_data["DATE"])
        else:
            print(f"Node {self.node_id} has empty dataset!")

        self.current_end_date = None
        if not self.full_data.empty:
            # The start date is where the training ends in the first round
            self.current_end_date = start_date
            self.data = self.full_data[self.full_data["DATE"] <= self.current_end_date]

        # Build the initial environment with data up to current_end_date
        env_name = f"ClimateEnv_{self.node_id}"
        register_env(env_name, lambda cfg: self._build_env())

        # PPO policy configuration
        self.config = (
            PPOConfig()
            .framework("torch")
            .training(gamma=0.99, lr=0.0003, train_batch_size=4000)
            .environment(env_name)
            .resources(num_gpus=0)  # CPU-based
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False
            )
        )

        # Trainer PPO
        self.trainer = self.config.build()
        print(f"Node {self.node_id} initialized.")


    def _build_env(self):
        """
        Build a ClimateEnv with data up to self.current_end_date.
        """
        if self.full_data.empty or self.current_end_date is None:
            return ClimateEnv(pd.DataFrame())  # empty environment

        # Filter rows where DATE <= current_end_date
        limited_data = self.full_data[self.full_data["DATE"] <= self.current_end_date].copy()
        # In case there's no row, pass an empty DataFrame
        if limited_data.empty:
            return ClimateEnv(pd.DataFrame())

        return ClimateEnv(limited_data)

    def reveal_new_data(self, days=1):
        """
        Advance current_end_date by 'days' days, so the environment sees more data.
        If data is exhausted, do nothing.
        """
        if self.full_data.empty:
            return  # no data to reveal

        max_date = self.full_data["DATE"].max()
        if self.current_end_date is None:
            self.current_end_date = self.full_data["DATE"].min()

        # Increment the end date by tot days
        self.current_end_date += pd.Timedelta(days=days)
        if self.current_end_date > max_date:
            self.current_end_date = max_date

        # Re-register environment
        env_name = f"ClimateEnv_{self.node_id}"
        register_env(env_name, lambda cfg: self._build_env())

        # Update the environment on all workers
        self.trainer.workers.foreach_worker(
            lambda w: setattr(w, "env_creator", lambda cfg: self._build_env())
        )

    def train(self, num_steps=1):
        """
        Train the PPO agent for a given number of steps.
        """
        for _ in range(num_steps):
            result = self.trainer.train()
            #print(f"Node {self.node_id} - Reward mean: {result}")

    def get_weights(self):
        """
        Return the weights of the local model for federated learning.
        """
        weights = self.trainer.get_weights()
        # Convert the weights to an ordered dictionary
        # These weights will be sent to the aggregator in the same order
        weights = collections.OrderedDict(weights)
        return weights

    def set_weights(self, global_weights):
        """
        Update the local model with the federated weights.
        """
        self.trainer.set_weights(global_weights)
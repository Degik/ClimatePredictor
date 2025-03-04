import os
import collections
# Ray
import ray
# RLlib
import ray.train
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
# Environment
from ClimateEnvironment import ClimateEnv
# Utilities
import pandas as pd
import time

@ray.remote(num_cpus=2)
class Node:
    def __init__(self, node_id, local_data_path, start_date, load_checkpoint=False, checkpoint_dir=None, train_batch_size=500):
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

        if checkpoint_dir is None:
            FileNotFoundError(f"[Node {self.node_id}] checkpoint_dir is not defined.")
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(checkpoint_dir, f"node_{self.node_id}")

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
        else:
            self.current_end_date = None

        # Create a unique environment name for this node
        env_name = f"ClimateEnv_{self.node_id}"


        # We neeed create a new ClimateEnv with the full dataset,
        # and we will call `update_end_date` later to limit it.
        def env_creator(cfg):
            env = ClimateEnv(self.full_data)
            if self.current_end_date is not None:
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
            .env_runners(num_env_runners=2)
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False
            )
            .training(
                gamma=0.995,
                lr=0.00001,
                train_batch_size=train_batch_size, #
                #sgd_minibatch_size=1024, # Deprecated
                num_sgd_iter=20,
                clip_param=0.2,
                vf_loss_coeff=0.5,
                vf_clip_param=10.0,
                entropy_coeff=0.01,
                lambda_=0.95,
                use_critic=True,
                use_gae=True,
            )
        )

        # Build the trainer
        self.trainer = self.config.build()

        # Load the latest checkpoint if requested
        if load_checkpoint:
            self.load_checkpoint()

        print(f"[Node {self.node_id}] Initialized with end date: {self.current_end_date}.")

    def load_checkpoint(self):
        """
        Load the latest checkpoint from disk.
        """
        try:
            self.trainer.restore(self.checkpoint_path) #https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm.Algorithm.restore.html
            print(f"[Node {self.node_id}] Loaded checkpoint from {self.checkpoint_path} !")
        except Exception as e:
            print(f"[Node {self.node_id}] No checkpoint found or failed to load: {e}")

    def save_checkpoint(self):
        """
        Save the latest checkpoint to disk.
        """
        try:
            os.makedirs(self.checkpoint_path, exist_ok=True)
            # Save the current checkpoint
            self.trainer.save_checkpoint(self.checkpoint_path)
            print(f"[Node {self.node_id}] Checkpoint saved to {self.checkpoint_path}")
        except Exception as e:
            print(f"[Node {self.node_id}] Failed to save checkpoint: {e}")


    def add_new_days(self, days=1):
        """
        Extend the current_end_date by the specified number of days and update the environment.
        """
        if self.full_data.empty:
            print(f"[Node {self.node_id}] No data available to reveal.")
            return

        max_date = self.full_data["DATE"].max()
        self.current_end_date = min(self.current_end_date + pd.Timedelta(days=days), max_date)

        new_end_date = self.current_end_date

        def do_update(env):
            env.update_end_date(new_end_date)

        # Update the environment
        # Note that: https://discuss.ray.io/t/correct-way-of-using-foreach-worker-and-foreach-env/21000
        # I've spent a lot time to resolve this issue
        # Now ray uses gymnasium, so we need to use env_runner_group and not workers
        self.trainer.env_runner_group.foreach_env(do_update)

        print(f"[Node {self.node_id}] Revealed new data up to: {new_end_date}.")
        return new_end_date

    def train(self, num_steps=1):
        """
        Train the PPO agent for a specified number of steps (iterations).
        
        Args:
            num_steps (int): Number of training iterations to run.
        
        Returns:
            tuple: Mean values of VF loss, policy loss, KL divergence, and entropy
        """
        mean_VF_loss = 0
        mean_policy_loss = 0
        mean_kl = 0
        mean_entropy = 0
        start_time = time.time()
        for i in range(num_steps):
            #sample_time = self.trainer.env_runner_group.foreach_env(lambda env: env.sample())
            #print(f"[Node {self.node_id}] Sample Time: {sample_time}")
            #
            result = self.trainer.train()
            # Logg
            policy_loss = result["info"]["learner"]["default_policy"]["learner_stats"]["policy_loss"]
            vf_loss = result["info"]["learner"]["default_policy"]["learner_stats"]["vf_loss"]
            kl = result["info"]["learner"]["default_policy"]["learner_stats"]["kl"]
            entropy = result["info"]["learner"]["default_policy"]["learner_stats"]["entropy"]
            
            num_steps_sampled = result["num_env_steps_sampled"]
            num_steps_trained = result["num_env_steps_trained"]
            training_iteration = result["training_iteration"]

            # Print training stats
            print(f"[Node {self.node_id}] Training Iteration {training_iteration}: "
                f"Steps Sampled: {num_steps_sampled}, Steps Trained: {num_steps_trained}")
            
            print(f"[Node {self.node_id}] Training Stats - Policy Loss: {policy_loss:.4f}, "
                f"VF Loss: {vf_loss:.4f}, KL: {kl:.4f}, Entropy: {entropy:.4f}")
            # Update the mean values
            mean_VF_loss += vf_loss
            mean_policy_loss += policy_loss
            mean_kl += kl
            mean_entropy += entropy
            self.metrics = { "VF_loss": mean_VF_loss, "policy_loss": mean_policy_loss, "kl": mean_kl, "entropy": mean_entropy }
            # Save the checkpoint
            self.save_checkpoint()

        # Calculate the mean values
        mean_VF_loss /= num_steps           #
        mean_policy_loss /= num_steps       #
        mean_kl /= num_steps                #
        mean_entropy /= num_steps           #
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[Node {self.node_id}] Training completed in {elapsed_time:.4f} seconds.")
        return mean_VF_loss, mean_policy_loss, mean_kl, mean_entropy, elapsed_time

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

    def ping(self):
        """
        Simple ping method to check if the node is alive.
        """
        return True
    
    def node_id(self):
        """
        Return the unique identifier for this node.
        """
        return self.node_id
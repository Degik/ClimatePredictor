import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ClimateEnvironment import ClimateEnv

@ray.remote
class Node:
    def __init__(self, node_id, name, data):
        self.node_id = node_id
        self.node_name = name
        # This need use a different dataset for each station
        self.env = ClimateEnv(data) # Local istance of the environment

        # PPO Configuration
        self.config = (
            PPOConfig()
            .framework("torch")
            .training(gamma=0.99, lr=0.0003, train_batch_size=4000)
            .rollouts(num_rollout_workers=1, num_envs_per_worker=1, remote_worker_envs=True)
            .resources(num_gpus=0)  # Nodes are CPU-based
        )

        # Initialize the trainer
        self.trainer = self.config.build(env=self.env)
        print(f"Node {self.node_id} initialized.")

    def train(self, num_steps=1):
        for _ in range(num_steps):
            result = self.trainer.train()
            print(f"Node {self.node_id} - Reward-mean: {result['episode_reward_mean']}")

    def get_weights(self):
        "This are essential for the Federated Learning, then we will use it for the aggregation"
        return self.trainer.get_policy().get_weights()

    def set_weights(self, global_weights):
        "Update the weights of the local model"
        self.trainer.get_policy().set_weights(global_weights)
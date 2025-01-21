import ray
import pandas as pd
#
from Node import Node
from FederatedAggregator import FederatedAggregator
#
import os

# Ray initialization
ray.init(address="auto", runtime_env={"working_dir": os.getcwd()})

nodes = [
    Node.options(resources={"n12": 1}).remote(node_id=0, local_data_path="1.csv"),
    Node.options(resources={"n13": 1}).remote(node_id=1, local_data_path="2.csv")
]

aggregator = FederatedAggregator.remote(nodes)

# Training federato
NUM_ROUNDS = 10
for i in range(NUM_ROUNDS):
    print(f"\n===== ROUND {i} =====")
    print("Training the nodes...")
    ray.get([node.train.remote(num_steps=1) for node in nodes])  # Train the nodes
    print("Nodes trained!")
    print("Aggregating the weights...")
    ray.get(aggregator.federated_averaging.remote())             # Aggregation of the weights
    print("Weights aggregated!")
    
ray.shutdown()
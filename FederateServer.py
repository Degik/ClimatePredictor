import ray
import pandas as pd
#
from Node import Node
from FederatedAggregator import FederatedAggregator
#
import os

# Ray initialization
ray.init(address="auto", runtime_env={"working_dir": os.getcwd()})
#ray.init(address="auto")
# Path to the local data
path_node = "/home/ubuntu/davide_b/ClimatePredictor_RL_FL/dataset/stations/"

nodes = [
    Node.options(resources={"n12": 1}).remote(node_id=0, local_data_path=path_node + "1.csv"),
    Node.options(resources={"n12": 1}).remote(node_id=1, local_data_path=path_node + "2.csv"),
    Node.options(resources={"n12": 1}).remote(node_id=2, local_data_path=path_node + "3.csv"),
    Node.options(resources={"n12": 1}).remote(node_id=3, local_data_path=path_node + "4.csv"),
    Node.options(resources={"n13": 1}).remote(node_id=4, local_data_path=path_node + "5.csv"),
    Node.options(resources={"n13": 1}).remote(node_id=5, local_data_path=path_node + "6.csv"),
    Node.options(resources={"n13": 1}).remote(node_id=6, local_data_path=path_node + "7.csv"),
    Node.options(resources={"n13": 1}).remote(node_id=7, local_data_path=path_node + "8.csv"),
    Node.options(resources={"n14": 1}).remote(node_id=8, local_data_path=path_node + "9.csv"),
    Node.options(resources={"n14": 1}).remote(node_id=9, local_data_path=path_node + "10.csv"),
    Node.options(resources={"n14": 1}).remote(node_id=10, local_data_path=path_node + "11.csv"),
    Node.options(resources={"n14": 1}).remote(node_id=11, local_data_path=path_node + "12.csv")
]

aggregator = FederatedAggregator.remote(nodes)

# Training loop
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
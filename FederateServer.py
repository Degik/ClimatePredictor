import ray
import pandas as pd
# RLlib
from Node import Node
from FederatedAggregator import FederatedAggregator
# Utilities
import os
import time

# Ray initialization
ray.init(address="auto", runtime_env={"working_dir": os.getcwd()})
#ray.init(address="auto")
# Path to the local data
path_node = "/home/ubuntu/davide_b/ClimatePredictor_RL_FL/datasets_hourly/"
start_date = pd.Timestamp("2022-01-01")
nodes = [
    Node.options(resources={"n12": 1}).remote(node_id=0, local_data_path=path_node + "1.csv", start_date=start_date),
    Node.options(resources={"n12": 1}).remote(node_id=1, local_data_path=path_node + "2.csv", start_date=start_date),
    Node.options(resources={"n12": 1}).remote(node_id=2, local_data_path=path_node + "3.csv", start_date=start_date),
    Node.options(resources={"n13": 1}).remote(node_id=3, local_data_path=path_node + "4.csv", start_date=start_date),
    Node.options(resources={"n13": 1}).remote(node_id=4, local_data_path=path_node + "5.csv", start_date=start_date),
    Node.options(resources={"n13": 1}).remote(node_id=5, local_data_path=path_node + "6.csv", start_date=start_date),
    Node.options(resources={"n14": 1}).remote(node_id=6, local_data_path=path_node + "7.csv", start_date=start_date),
    Node.options(resources={"n14": 1}).remote(node_id=7, local_data_path=path_node + "8.csv", start_date=start_date),
    Node.options(resources={"n14": 1}).remote(node_id=8, local_data_path=path_node + "9.csv", start_date=start_date),
]

aggregator = FederatedAggregator.remote(nodes)

round_count = 0
while True:
    print(f"=== Round {round_count} ===")
    print("Revealing new data started.")
    ray.get([node.add_new_days.remote(days=1) for node in nodes])
    print("Revealing new data completed.")

    # Training
    print("Training started.")
    ray.get([node.train.remote(num_steps=1) for node in nodes])
    print("Training completed.")

    # Federated averaging
    print("Federated averaging started.")
    ray.get(aggregator.federated_averaging.remote())
    print("Federated averaging completed.")

    # Set the global weights on each node
    print("Setting global weights.")
    global_weights = ray.get(aggregator.get_global_weights.remote())
    for node in nodes:
        node.set_weights.remote(global_weights)
    print("Global weights set.")

    # (Opzionale) Evaluate
    # scores = ray.get([node.evaluate.remote() for node in nodes])
    # mean_score = sum(scores) / len(scores)
    # print(f"Mean local eval: {mean_score}")

    round_count += 1

    # Wait for a while before starting the next round
    time.sleep(10)


#ray.shutdown()
import ray
import pandas as pd
import ray.exceptions
# RLlib
from Node import Node
from FederatedAggregator import FederatedAggregator
# Utilities
import os
import time
import warnings
# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# GLOBAL CONFIGURATION
####################################################################################################
# Exceptions list
EXCEPTIONS = tuple([
    ray.exceptions.RayActorError,       # https://docs.ray.io/en/latest/ray-core/api/doc/ray.exceptions.RayActorError.html#ray.exceptions.RayActorError
    ray.exceptions.GetTimeoutError,     # https://docs.ray.io/en/latest/ray-core/api/doc/ray.exceptions.GetTimeoutError.html#ray.exceptions.GetTimeoutError
    ray.exceptions.ActorDiedError,      # https://docs.ray.io/en/latest/ray-core/api/doc/ray.exceptions.ActorDiedError.html#ray.exceptions.ActorDiedError
    ray.exceptions.RayTaskError         # https://docs.ray.io/en/latest/ray-core/api/doc/ray.exceptions.RayTaskError.html#ray.exceptions.RayTaskError
])
# All exceptions can be found here: https://docs.ray.io/en/latest/ray-core/api/exceptions.html#ray-core-exceptions

# Path where to save the checkpoints
CHECKPOINT_DIR = "/home/ubuntu/davide_b/checkpoints/"
# Path to the local data
PATH_NODE = "/home/ubuntu/davide_b/ClimatePredictor_RL_FL/datasets_hourly/"
# The ending date for the training data
START_DATE = pd.Timestamp("2024-01-01")
#
CSV_FILES = [ "1.csv", "2.csv", "3.csv", "4.csv", "5.csv", "6.csv", "7.csv", "8.csv", "9.csv" ]
####################################################################################################


# NODE CONFIGURATION
####################################################################################################
# Ray initialization
ray.init(address="auto", runtime_env={"working_dir": os.getcwd()})


""" nodes = [
    Node.options(resources={"n12": 2}).remote(node_id=0, local_data_path=PATH_NODE + "1.csv", start_date=START_DATE, checkpoint_dir=CHECKPOINT_DIR, load_checkpoint=True),
    Node.options(resources={"n12": 2}).remote(node_id=1, local_data_path=PATH_NODE + "2.csv", start_date=START_DATE, checkpoint_dir=CHECKPOINT_DIR, load_checkpoint=True),
    Node.options(resources={"n12": 2}).remote(node_id=2, local_data_path=PATH_NODE + "3.csv", start_date=START_DATE, checkpoint_dir=CHECKPOINT_DIR, load_checkpoint=True),
    Node.options(resources={"n13": 2}).remote(node_id=3, local_data_path=PATH_NODE + "4.csv", start_date=START_DATE, checkpoint_dir=CHECKPOINT_DIR, load_checkpoint=True),
    Node.options(resources={"n13": 2}).remote(node_id=4, local_data_path=PATH_NODE + "5.csv", start_date=START_DATE, checkpoint_dir=CHECKPOINT_DIR, load_checkpoint=True),
    Node.options(resources={"n13": 2}).remote(node_id=5, local_data_path=PATH_NODE + "6.csv", start_date=START_DATE, checkpoint_dir=CHECKPOINT_DIR, load_checkpoint=True),
    Node.options(resources={"n14": 2}).remote(node_id=6, local_data_path=PATH_NODE + "7.csv", start_date=START_DATE, checkpoint_dir=CHECKPOINT_DIR, load_checkpoint=True),
    Node.options(resources={"n14": 2}).remote(node_id=7, local_data_path=PATH_NODE + "8.csv", start_date=START_DATE, checkpoint_dir=CHECKPOINT_DIR, load_checkpoint=True),
    Node.options(resources={"n14": 2}).remote(node_id=8, local_data_path=PATH_NODE + "9.csv", start_date=START_DATE, checkpoint_dir=CHECKPOINT_DIR, load_checkpoint=True),
]
 """
nodes = []
for i, dataset in enumerate(CSV_FILES):
    node_handle = Node.options(
        resources={f"n{i // 3 + 12}": 2}).remote(
            node_id=i, 
            local_data_path=PATH_NODE + dataset, 
            start_date=START_DATE, 
            checkpoint_dir=CHECKPOINT_DIR, 
            load_checkpoint=True
        )
    nodes.append((i, node_handle))

# Create the Federated Aggregator
aggregator = FederatedAggregator.options(resources={"head": 1}).remote(nodes=nodes, EXCEPTIONS=EXCEPTIONS)
####################################################################################################

# MAIN LOOP
####################################################################################################
# Initialize the active and failed nodes
active_nodes = set(nodes) # All nodes are active
failed_nodes = set()      # Used for the retry mechanism

timeout = 15
end_date = None
# Main loop
round_count = 0
while True:
    print(f"=== Round {round_count} ===")

    # Metrics lists
    mean_VF_loss_l, mean_policy_loss_l, mean_kl_l, mean_entropy_l = [], [], [], []
    #

    # Check if there are any failed nodes
    # If a node is offline, try to reconnect it
    for (node_id, node_handle) in list(failed_nodes):
        try:
            ray.get(node_handle.ping.remote(), timeout=timeout)
            print(f"[HEAD][INFO] Node {node_id} reconnected!")
            failed_nodes.remove((node_id, node_handle))
            active_nodes.add((node_id, node_handle))
        except EXCEPTIONS:
            print(f"[HEAD][WARN] Node {node_id} is still offline. Trying to create a new node...")
            try:
                resource_id = "n" + str(node_id // 3 + 12)
                local_data_path = PATH_NODE + f"{node_id + 1}.csv"
                
                # Create a new node with the same ID
                new_node_handle = Node.options(resources={resource_id: 2}).remote(
                    node_id=node_id, 
                    local_data_path=local_data_path, 
                    start_date=START_DATE, 
                    checkpoint_dir=CHECKPOINT_DIR, 
                    load_checkpoint=True
                )
                # Check if the node is active
                ray.get(new_node_handle.ping.remote(), timeout=timeout)

                print(f"[HEAD][INFO] Node {node_id} recreated successfully.")
                # Update active_nodes and failed_nodes
                active_nodes.add((node_id, new_node_handle))
                failed_nodes.remove((node_id, node_handle))
            except EXCEPTIONS as e:
               print(f"[HEAD][WARN] Failed to recreate the node {node_id}, there's not enough resources. Error: {e}")

    print(f"[HEAD][INFO] Active Nodes: {len(active_nodes)} | Failed Nodes: {len(failed_nodes)}")

    print("[HEAD][INFO] Revealing new data...")
    for (node_id, node_handle) in list(active_nodes):
        try:
            end_date = ray.get(node_handle.add_new_days.remote(days=1), timeout=timeout)
        except EXCEPTIONS as e:
            print(f"[HEAD][WARN] Node {node_id} did not respond. Marking as failed. Error: {e}")
            active_nodes.remove((node_id, node_handle))
            failed_nodes.add((node_id, node_handle))
    print("[HEAD][INFO] Data revealed.")

    #Check if there is any active node
    if not active_nodes:
        print("[HEAD][WARN] No active nodes available. Skipping round.")
        time.sleep(10)
        continue

    # Training
    print("[HEAD][INFO] Training started.")
    for (node_id, node_handle) in list(active_nodes):
        try:
            mean_VF_loss, mean_policy_loss, mean_kl, mean_entropy = ray.get(node_handle.train.remote(num_steps=1), timeout=timeout)
            mean_VF_loss_l.append(mean_VF_loss)
            mean_policy_loss_l.append(mean_policy_loss)
            mean_kl_l.append(mean_kl)
            mean_entropy_l.append(mean_entropy)
        except EXCEPTIONS as e:
            print(f"[HEAD][WARN] Node {node_id} failed during training. Marking as failed. Error: {e}")
            active_nodes.remove((node_id, node_handle))
            failed_nodes.add((node_id, node_handle))
    print("[HEAD][INFO] Training completed.")

    # Federated averaging
    print("[HEAD][INFO] Federated averaging started.")
    try:
        global_weights = ray.get(aggregator.federated_averaging.remote(), timeout=timeout)
        print("[HEAD][INFO] Federated averaging completed.")
    except EXCEPTIONS as e:
        print(f"[HEAD][WARN] Federated Aggregator is not responding! Skipping this iteration. Error: {e}")
        time.sleep(10)
        continue

    # Set the global weights on each node
    print("[HEAD][INFO] Setting global weights.")
    for (node_id, node_handle) in list(active_nodes):
        try:
            node_handle.set_weights.remote(global_weights)
        except EXCEPTIONS as e:
            print(f"[HEAD][WARN] Unable to update weights for node {node_id}. Error: {e}")
    print("[HEAD][INFO] Global weights set.")

    # Print the mean metrics for this round
    if mean_VF_loss_l:
        print(f"[HEAD][INFO] Mean VF Loss: {sum(mean_VF_loss_l) / len(mean_VF_loss_l)}")
    if mean_policy_loss_l:
        print(f"[HEAD][INFO] Mean Policy Loss: {sum(mean_policy_loss_l) / len(mean_policy_loss_l)}")
    if mean_kl_l:
        print(f"[HEAD][INFO] Mean KL: {sum(mean_kl_l) / len(mean_kl_l)}")
    if mean_entropy_l:
        print(f"[HEAD][INFO] Mean Entropy: {sum(mean_entropy_l) / len(mean_entropy_l)}")

    round_count += 1

    # Wait for a while before starting the next round
    time.sleep(10)
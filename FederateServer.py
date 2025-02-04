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
START_DATE = pd.Timestamp("2025-01-01")
#
CSV_FILES = [ "1.csv", "2.csv", "3.csv", "4.csv", "5.csv", "6.csv", "7.csv", "8.csv", "9.csv" ]
####################################################################################################


# NODE CONFIGURATION
####################################################################################################
start_time = time.time()
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
            load_checkpoint=True,
            train_batch_size=500
        )
    nodes.append((i, node_handle))

# Create the Federated Aggregator
aggregator = FederatedAggregator.options(resources={"head": 1}).remote(nodes=nodes, timeout=30, EXCEPTIONS=EXCEPTIONS)
time_init = time.time() - start_time
print(f"[HEAD][INFO] Initialization completed in {time_init:.4f} seconds.")
####################################################################################################

# MAIN LOOP
####################################################################################################
# Initialize the active and failed nodes
active_nodes = set(nodes) # All nodes are active
failed_nodes = set()      # Used for the retry mechanism

num_step = 20
timeout = 60
end_date = None
# Main loop
round_count = 0
while True:
    print(f"=== Round {round_count} ===")

    # METRICS VARIABLES
    ####################################################################################################
    # Metrics lists
    mean_VF_loss_l, mean_policy_loss_l, mean_kl_l, mean_entropy_l = [], [], [], []
    # Times list where will be stored the training time for each node
    times = {}
    # Computation times
    time_connect = None
    time_total = None
    time_train = None
    time_rvl = None
    time_aggr = None
    #
    start_time_total = time.time()
    ####################################################################################################


    # RECONNECT NODES
    # TRY TO RECONNECT FAILED NODES AND CREATE NEW NODES IF NEEDED
    ####################################################################################################
    # Start time
    start_time = time.time()

    ### Ping the failed nodes
    ping_tasks = {}
    # Crate a ping task for each failed node
    for (node_id, node_handle) in list(failed_nodes):
        task = node_handle.ping.remote()
        ping_tasks[task] = (node_id, node_handle)

    # Wait for the ping tasks to complete
    done, not_done = ray.wait(
        list(ping_tasks.keys()),
        timeout=timeout,
        num_returns=len(ping_tasks)
    )

    ### Nodes to recreate
    nodes_to_recreate = []
    ### Check the results
    for task in done:
        node_id, node_handle = ping_tasks[task]
        try:
            # If the ping is successful, the node is back online
            ray.get(task)
            print(f"[HEAD][INFO] Node {node_id} reconnected!")
            failed_nodes.remove((node_id, node_handle))
            active_nodes.add((node_id, node_handle))
        except EXCEPTIONS as e:
            print(f"[HEAD][WARN] Node {node_id} ping failed: {e}. Trying to recreate node.")
            nodes_to_recreate.append((node_id, node_handle))
    
    # Take the ping failed nodes and try to recreate them
    for task in not_done:
        node_id, node_handle = ping_tasks[task]
        print(f"[HEAD][WARN] Node {node_id} did not respond in time. Trying to recreate node.")
        nodes_to_recreate.append((node_id, node_handle))

    recreation_tasks = {}
    # Create a new node for each failed node
    for (node_id, old_node_handle) in nodes_to_recreate:
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
        new_ping = new_node_handle.ping.remote()
        recreation_tasks[new_ping] = (node_id, old_node_handle, new_node_handle)

    done_recreation, not_done_recreation = ray.wait(
        list(recreation_tasks.keys()),
        timeout=timeout,
        num_returns=len(recreation_tasks)
    )

    # Check the recreation tasks that completed
    for task in done_recreation:
        node_id, old_node_handle, new_node_handle = recreation_tasks[task]
        try:
            ray.get(task)
            print(f"[HEAD][INFO] Node {node_id} recreated successfully.")
            active_nodes.add((node_id, new_node_handle))
            failed_nodes.remove((node_id, old_node_handle))
        except EXCEPTIONS as e:
            print(f"[HEAD][WARN] Failed to recreate node {node_id}: {e}")

    # Check the recreation tasks that did not complete
    for task in not_done_recreation:
        node_id, old_node_handle, new_node_handle = recreation_tasks[task]
        print(f"[HEAD][WARN] Node {node_id} recreation timed out.")

    time_connect = time.time() - start_time

    print(f"[HEAD][INFO] Active Nodes: {len(active_nodes)} | Failed Nodes: {len(failed_nodes)}")

    # REVEAL NEW DATA TO THE NODES
    # ADD A NEW DAY TO THE TRAINING DATA
    ####################################################################################################
    start_time = time.time()

    print("[HEAD][INFO] Revealing new data...")
    add_day_tasks = {}
    for (node_id, node_handle) in list(active_nodes):
        task = node_handle.add_new_days.remote(days=1)
        add_day_tasks[task] = (node_id, node_handle)
    
    done, not_done = ray.wait(
        list(add_day_tasks.keys()),
        timeout=timeout,
        num_returns=len(add_day_tasks)
    )
    
    for task in done:
        node_id, node_handle = add_day_tasks[task]
        try:
            ray.get(task)
            print(f"[HEAD][INFO] Node {node_id} received new data.")
        except EXCEPTIONS as e:
            print(f"[HEAD][WARN] Node {node_id} failed to receive new data. Error: {e}")
            active_nodes.remove((node_id, node_handle))
            failed_nodes.add((node_id, node_handle))
    
    print("[HEAD][INFO] Data revealed.")
    time_rvl = time.time() - start_time
    ####################################################################################################

    #Check if there is any active node
    if not active_nodes:
        print("[HEAD][WARN] No active nodes available. Skipping round.")
        time.sleep(10)
        continue
    
    # TRAINING
    # START THE TRAINING ON EACH NODE
    ####################################################################################################
    start_time = time.time()
    print("[HEAD][INFO] Training started.")

    try:
        # Start the training on each node
        training_tasks = {}
        for (node_id, node_handle) in list(active_nodes):
            # Start the training on the node
            task = node_handle.train.remote(num_steps=num_step)
            # Store the task
            training_tasks[task] = (node_id, node_handle)
        
        # Wait for the training to complete
        done, not_done = ray.wait(list(training_tasks.keys()), 
                                  timeout=timeout,
                                  num_returns=len(training_tasks.keys()))
    except EXCEPTIONS as e:
        print(f"[HEAD][WARN] Training failed. Error: {e}")
        continue

    # Get the results
    for task in done:
        node_id, node_handle = training_tasks[task]
        try:
            mean_VF_loss, mean_policy_loss, mean_kl, mean_entropy, elapsed_time = ray.get(task)
            mean_VF_loss_l.append(mean_VF_loss)
            mean_policy_loss_l.append(mean_policy_loss)
            mean_kl_l.append(mean_kl)
            mean_entropy_l.append(mean_entropy)
            times[node_id] = elapsed_time
        except EXCEPTIONS as e:
            print(f"[HEAD][WARN] Node {node_id} failed during training. Marking as failed. Error: {e}")
            active_nodes.remove((node_id, node_handle))
            failed_nodes.add((node_id, node_handle))

    for task in not_done:
        node_id, node_handle = training_tasks[task]
        print(f"[HEAD][WARN] Node {node_id} did not respond. Marking as failed.")
        active_nodes.remove((node_id, node_handle))
        failed_nodes.add((node_id, node_handle))

    print("[HEAD][INFO] Training completed.")
    time_train = time.time() - start_time
    ####################################################################################################

    # AGGREGATION
    # FEDERATED AVERAGING OF THE WEIGHTS
    ####################################################################################################
    start_time = time.time()
    # Federated averaging
    print("[HEAD][INFO] Federated averaging started.")
    try:
        global_weights = ray.get(aggregator.federated_averaging.remote(), timeout=timeout)
        print("[HEAD][INFO] Federated averaging completed.")
    except EXCEPTIONS as e:
        print(f"[HEAD][WARN] Federated Aggregator is not responding! Skipping this iteration. Error: {e}")
        time.sleep(10)
        continue

    # UPDATE THE GLOBAL WEIGHTS
    ####################################################################################################
    # Set the global weights on each node
    print("[HEAD][INFO] Setting global weights.")
    # Create a task for each node
    update_weights_tasks = {}
    for (node_id, node_handle) in list(active_nodes):
        task = node_handle.set_weights.remote(global_weights)
        update_weights_tasks[task] = (node_id, node_handle)
    
    # Wait for the tasks to complete
    done, not_done = ray.wait(
        list(update_weights_tasks.keys()),
        timeout=timeout,
        num_returns=len(update_weights_tasks)
    )
    # Check the results
    for task in done:
        node_id, node_handle = update_weights_tasks[task]
        try:
            ray.get(task)
            print(f"[HEAD][INFO] Global weights set on node {node_id}.")
        except EXCEPTIONS as e:
            print(f"[HEAD][WARN] Unable to update weights for node {node_id}. Error: {e}")
            active_nodes.remove((node_id, node_handle))
            failed_nodes.add((node_id, node_handle))

    print("[HEAD][INFO] Global weights set.")
    time_aggr = time.time() - start_time
    ####################################################################################################

    # PRINT METRICS
    ####################################################################################################
    # Print the mean metrics for this round
    if mean_VF_loss_l:
        print(f"[HEAD][INFO] Mean VF Loss: {sum(mean_VF_loss_l) / len(mean_VF_loss_l)}")
    if mean_policy_loss_l:
        print(f"[HEAD][INFO] Mean Policy Loss: {sum(mean_policy_loss_l) / len(mean_policy_loss_l)}")
    if mean_kl_l:
        print(f"[HEAD][INFO] Mean KL: {sum(mean_kl_l) / len(mean_kl_l)}")
    if mean_entropy_l:
        print(f"[HEAD][INFO] Mean Entropy: {sum(mean_entropy_l) / len(mean_entropy_l)}")

    # Print computation times
    for node_id, time_n in times.items():
        print(f"[HEAD][INFO] Node {node_id} training time: {time_n:.4f} seconds")
    # Print the mean time
    if times:
        print(f"[HEAD][INFO] Mean Training Time: {sum(times) / len(times):.4f} seconds")
    
    # Print the computation times
    print(f"[HEAD][INFO] Connection Time: {time_connect:.4f} seconds")
    print(f"[HEAD][INFO] Data Reveal Time: {time_rvl:.4f} seconds")
    print(f"[HEAD][INFO] Training Time: {time_train:.4f} seconds")
    print(f"[HEAD][INFO] Aggregation Time: {time_aggr:.4f} seconds")
    # Print the total time
    time_total = time.time() - start_time_total
    print(f"[HEAD][INFO] Total Time: {time_total:.4f} seconds")
    ####################################################################################################
    round_count += 1

    # Wait for a while before starting the next round
    time.sleep(10)
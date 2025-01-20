import ray
import pandas as pd
import Node
import FederatedAggregator

# Ray initialization
ray.init(ignore_reinit_error=True)

# Load climate date
# The datasets were located in dastaets folder
# Take all csv files and send to the nodes 1 to 1
# Take the csv file name and use as the node name

server = FederatedAggregator(nodes)

NUM_ROUNDS = 10

for i in range(NUM_ROUNDS):
    print(f"\n===== ROUND {i} =====")
    
    
ray.shutdown()
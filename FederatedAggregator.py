import ray
import numpy as np

@ray.remote
class FederatedAggregator:
    def __init__(self, nodes):
        # Take all nodes list, with different configuration for each station
        self.nodes = nodes

    def federated_averaging(self):
        """
        Aggregation of the weights of all nodes using Federated Averaging.
        """
        global_weights = self.nodes[0].get_weights()

        for key in global_weights.keys():
            global_weights[key] = np.mean(
                [node.get_weights()[key] for node in self.nodes], axis=0
            )

        # Update all the nodes with the fedarated weights
        for node in self.nodes:
            node.set_weights(global_weights)
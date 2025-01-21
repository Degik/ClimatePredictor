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
        # Get the global weights
        global_weights = ray.get(self.nodes[0].get_weights.remote())
        print('Global weights:', global_weights)

        # Average the weights of all nodes
        for key in global_weights.keys():
            global_weights[key] = np.mean(
                [ray.get(node.get_weights.remote())[key] for node in self.nodes], axis = 0
            )
        print('Global weights after averaging:', global_weights)

        print('Updating the nodes...')
        # Update all the nodes with the fedarated weights
        for node in self.nodes:
            node.set_weights.remote(global_weights)
        print('Nodes updated!')
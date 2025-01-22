# Ray
import ray
# Utilities
import collections
# NumPy
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
        weights_list = ray.get([node.get_weights.remote() for node in self.nodes])

        if not all(isinstance(weights, collections.OrderedDict) for weights in weights_list):
            raise TypeError("All weights should be of type OrderedDict!")
        
        # Average the weights of all nodes
        averaged_weights = collections.OrderedDict()
        for key in weights_list[0].keys():
            averaged_weights[key] = np.mean([w[key] for w in weights_list], axis=0)
        print('Weights averaged!')

        print('Updating the nodes...')
        # Update all the nodes with the federated weights
        ray.get([node.set_weights.remote(averaged_weights) for node in self.nodes])
        print('Nodes updated!')
        
        return averaged_weights
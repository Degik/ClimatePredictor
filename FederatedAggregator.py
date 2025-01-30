# Ray
import ray
# Utils
import collections
# NumPy and PyTorch
import numpy as np
import ray.exceptions
import torch

def _is_numeric(x):
    """Return True if x is a numeric type (int, float, np.ndarray, torch.Tensor)."""
    if isinstance(x, (int, float)):
        return True
    if isinstance(x, np.ndarray):
        # Check if the dtype is numeric
        return x.dtype.kind in ("i", "f")  # int or float
    if isinstance(x, torch.Tensor):
        # Check if the dtype is numeric
        return x.dtype in (torch.float16, torch.float32, torch.float64,
                           torch.int16, torch.int32, torch.int64)
    return False

def _to_tensor(x):
    """ Convert x to a torch.Tensor. """
    if isinstance(x, (int, float)):
        return torch.tensor(x, dtype=torch.float32)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    elif isinstance(x, torch.Tensor):
        return x.float()
    else:
        raise TypeError(f"[HEAD][WARN] Implicit conversion to tensor not supported for type {type(x)}.")

def _recursive_average(dict_list):
    """
    From a list of dictionaries, compute the average of all numeric fields.
    If a field is a dictionary, recursively compute the average of its fields.
    """
    out = collections.OrderedDict()

    all_keys = set()
    for d in dict_list:
        all_keys.update(d.keys())
    
    for key in sorted(all_keys):
        # Collect all values for this key
        vals = [d[key] for d in dict_list if key in d]
        
        # IF all vals are dictionaries -> recursively average
        if all(isinstance(v, dict) for v in vals):
            out[key] = _recursive_average(vals)
        
        # If all vals are numeric -> convert to tensor and check shape
        elif all(_is_numeric(v) for v in vals):
            # Convert to tensor
            ts = [_to_tensor(v) for v in vals]
            shapes = [t.shape for t in ts]
            if not all(s == shapes[0] for s in shapes):
                # If shapes are different, skip
                print(f"[HEAD][SKIP] Key {key}: different shapes.")
                continue
            # Calculate means
            stacked = torch.stack(ts, dim=0)
            mean_t = stacked.mean(dim=0)
            out[key] = mean_t
        
        else:
            print(f"[HEAD][SKIP] Key {key}: not all numeric or all dictionaries.")
            pass
    
    return out

@ray.remote
class FederatedAggregator:
    def __init__(self, nodes, EXCEPTIONS=None):
        self.nodes = set(nodes)
        self.failed_nodes = set()
        self.EXCEPTIONS = EXCEPTIONS

    def federated_averaging(self):
        """
        Take the weights from each node, average them, and set the average as the new global weights.
        In RLlib PPO, the actual weights are stored in weights["default_policy"].
        """
        # Take the weights from each node
        weights_list = []
        successful_nodes = set()

        for (node_id, node_handle) in list(self.nodes):
            try:
                weights = ray.get(node_handle.get_weights.remote())
                weights_list.append(weights)
                successful_nodes.add((node_id, node_handle))
            except self.EXCEPTIONS as e:
                print(f"[HEAD][WARN] Node {node_id} failed to return weights. Error: {e}")
                self.nodes.remove((node_id, node_handle))
                self.failed_nodes.add((node_id, node_handle))

        print(f"[HEAD][INFO] Weights collected from {len(successful_nodes)} nodes.")

        if not weights_list:
            print("[HEAD][WARN] No weights collected. Skipping aggregation.")
            return None

        # Average the weights
        default_policy_list = []
        for w in weights_list:
            # Check if the node has a "default_policy" field
            if "default_policy" in w:
                default_policy_list.append(w["default_policy"])
            else:
                print("[HEAD][WARN] Nothing to aggregate in this node.")
        
        if not default_policy_list:
            # If no default_policy found, skip
            print("[HAED][WARN] No default_policy found in any node.")
            return

        # Recursive average
        avg_default_policy = _recursive_average(default_policy_list)
        
        # Return the new global weights (only the "default_policy" field)
        global_weights = collections.OrderedDict({
            "default_policy": avg_default_policy
        })

        print("[HAED][INFO] FedAvg: Global weights updated.")
        return global_weights

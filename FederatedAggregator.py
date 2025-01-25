# Ray
import ray
# Utils
import collections
# NumPy and PyTorch
import numpy as np
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
        raise TypeError(f"Implicit conversion to tensor not supported for type {type(x)}.")

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
                print(f"[SKIP] Key {key}: different shapes.")
                continue
            # Calculate means
            stacked = torch.stack(ts, dim=0)
            mean_t = stacked.mean(dim=0)
            out[key] = mean_t
        
        else:
            print(f"[SKIP] Key {key}: not all numeric or all dictionaries.")
            pass
    
    return out

@ray.remote
class FederatedAggregator:
    def __init__(self, nodes):
        self.nodes = nodes

    def federated_averaging(self):
        """
        Take the weights from each node, average them, and set the average as the new global weights.
        In RLlib PPO, the actual weights are stored in weights["default_policy"].
        """
        # Take the weights from each node
        weights_list = ray.get([node.get_weights.remote() for node in self.nodes])
        
        # Average the weights
        default_policy_list = []
        for w in weights_list:
            # Check if the node has a "default_policy" field
            if "default_policy" in w:
                default_policy_list.append(w["default_policy"])
            else:
                print("[WARN] Nothing to aggregate in this node.")
        
        if not default_policy_list:
            # If no default_policy found, skip
            print("[WARN] No default_policy found in any node.")
            return

        # Recursive average
        avg_default_policy = _recursive_average(default_policy_list)
        
        # Return the new global weights (only the "default_policy" field)
        global_weights = collections.OrderedDict({
            "default_policy": avg_default_policy
        })
        
        # Update the weights on each node
        ray.get([node.set_weights.remote(global_weights) for node in self.nodes])

        print("FedAvg: Global weights updated.")
        return global_weights

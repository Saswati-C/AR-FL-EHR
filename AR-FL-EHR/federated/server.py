
from typing import List, Dict
import torch

def fedavg(state_dicts: List[Dict[str, torch.Tensor]], weights: List[float] | None = None):
    if weights is None:
        weights = [1.0/len(state_dicts)] * len(state_dicts)
    out = {}
    for k in state_dicts[0].keys():
        out[k] = sum(w * sd[k] for w, sd in zip(weights, state_dicts))
    return out

class Server:
    def __init__(self, global_model: torch.nn.Module, device: str = 'cpu'):
        self.global_model = global_model.to(device)
        self.device = device

    def aggregate(self, client_states: List[Dict[str, torch.Tensor]], weights: List[float] | None = None):
        agg = fedavg(client_states, weights=weights)
        self.global_model.load_state_dict(agg)
        return agg

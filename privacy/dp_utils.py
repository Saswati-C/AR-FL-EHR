
from typing import List, Dict
import torch

def clip_and_add_noise(state_dicts: List[Dict[str, torch.Tensor]], clip_norm: float = 1.0, noise_multiplier: float = 0.0):
    clipped = []
    for sd in state_dicts:
        # Compute global norm
        total_sq = 0.0
        for p in sd.values():
            total_sq += float((p**2).sum())
        total_norm = (total_sq ** 0.5) + 1e-12
        scale = min(1.0, clip_norm / total_norm)
        new_sd = {k: v * scale for k, v in sd.items()}
        clipped.append(new_sd)

    if noise_multiplier > 0.0:
        for sd in clipped:
            for k in sd:
                sd[k] = sd[k] + torch.randn_like(sd[k]) * noise_multiplier
    return clipped


import torch
import torch.nn as nn

class DomainAwareAttention(nn.Module):
    """Simple feature-wise attention. Optionally per-client parameters."""
    def __init__(self, in_features: int, hidden: int = 64, per_client: bool = False, num_clients: int = 1):
        super().__init__()
        self.per_client = per_client
        if per_client:
            self.blocks = nn.ModuleList([nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, in_features),
                nn.Sigmoid()
            ) for _ in range(num_clients)])
        else:
            self.block = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, in_features),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor, client_id: int | None = None):
        if self.per_client:
            assert client_id is not None, "client_id required when per_client=True"
            weights = self.blocks[client_id](x)
        else:
            weights = self.block(x)
        return x * weights, weights

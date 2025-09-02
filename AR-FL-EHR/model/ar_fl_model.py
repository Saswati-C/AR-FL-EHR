
import torch
import torch.nn as nn
from .attention import DomainAwareAttention

class ARFLModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 2, use_attention: bool = True,
                 per_client: bool = False, num_clients: int = 1):
        super().__init__()
        self.use_attention = use_attention
        self.att = None
        if use_attention:
            self.att = DomainAwareAttention(in_features, per_client=per_client, num_clients=num_clients)

        self.backbone = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor, client_id: int | None = None):
        if self.use_attention and self.att is not None:
            x, attw = self.att(x, client_id)
        else:
            attw = None
        logits = self.backbone(x)
        return logits, attw

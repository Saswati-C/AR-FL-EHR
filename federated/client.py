
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from .adversarial_training import fgsm_attack, pgd_attack

class SimpleTensorDataset(data.Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X, self.y = X, y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class Client:
    def __init__(self, client_id: int, model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                 batch_size: int = 64, device: str = 'cpu'):
        self.client_id = client_id
        self.model = model
        self.device = device
        self.dataset = SimpleTensorDataset(X, y)
        self.loader = data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def local_train(self, epochs: int, lr: float, epsilon: float, pgd_steps: int, step_size: float,
                    lambda_adv: float = 0.5) -> Dict[str, torch.Tensor]:
        self.model.train()
        self.model.to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        for _ in range(epochs):
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                # mix clean + adversarial
                xb_adv = pgd_attack(self.model, xb, yb, epsilon=epsilon, steps=pgd_steps, step_size=step_size)
                logits_clean, _ = self.model(xb, client_id=self.client_id)
                logits_adv, _ = self.model(xb_adv, client_id=self.client_id)

                loss = F.cross_entropy(logits_clean, yb) + lambda_adv * F.cross_entropy(logits_adv, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        # Return state dict copy on CPU
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

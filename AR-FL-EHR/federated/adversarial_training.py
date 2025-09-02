
import torch
import torch.nn.functional as F

@torch.no_grad()
def _grad_sign(model, x, y, loss_fn):
    x = x.clone().detach().requires_grad_(True)
    logits, _ = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    return x.grad.data.sign()

def fgsm_attack(model, x, y, epsilon=0.03, loss_fn=None):
    if loss_fn is None:
        loss_fn = F.cross_entropy
    sign = _grad_sign(model, x, y, loss_fn)
    x_adv = (x + epsilon * sign).clamp(0.0, 1.0)
    return x_adv

def pgd_attack(model, x, y, epsilon=0.03, steps=10, step_size=0.003, loss_fn=None):
    if loss_fn is None:
        loss_fn = F.cross_entropy
    x_adv = x.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = x_adv.clamp(0.0, 1.0).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits, _ = model(x_adv)
        loss = loss_fn(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.sign()
        # project back to epsilon-ball
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = x_adv.clamp(0.0, 1.0).detach()
    return x_adv

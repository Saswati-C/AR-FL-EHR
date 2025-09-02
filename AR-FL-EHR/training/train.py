
"""Main training loop for AR-FL-EHR.

This is a *minimal* scaffold that simulates clients with random tensors so you can
verify end-to-end flow before wiring real EHR loaders.

Usage:
    python training/train.py --config experiments/configs/arfl_config.json
"""
import os, json, argparse, random
import torch
import torch.nn.functional as F

from model.ar_fl_model import ARFLModel
from federated.client import Client
from federated.server import Server
from privacy.dp_utils import clip_and_add_noise
from privacy.secure_aggregation import secure_aggregate

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def make_synthetic_clients(cfg: dict):
    num_clients = cfg.get('num_clients', 3)
    input_dim = cfg.get('input_dim', 256)
    num_classes = cfg.get('num_classes', 2)
    per_client = True

    clients = []
    for cid in range(num_clients):
        X = torch.rand(512, input_dim)
        y = torch.randint(0, num_classes, (512,))
        model = ARFLModel(in_features=input_dim, num_classes=num_classes, use_attention=True,
                          per_client=per_client, num_clients=num_clients)
        clients.append(Client(cid, model, X, y, batch_size=cfg['batch_size']))
    return clients

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to JSON config')
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build clients and a server with a global model of same shape
    clients = make_synthetic_clients(cfg)
    global_model = ARFLModel(cfg['input_dim'], cfg['num_classes'], use_attention=True,
                             per_client=True, num_clients=cfg['num_clients'])
    server = Server(global_model, device=device)

    rounds = cfg['rounds']
    for r in range(1, rounds+1):
        client_states = []
        for client in clients:
            # load latest global weights
            client.model.load_state_dict(server.global_model.state_dict())
            sd = client.local_train(
                epochs=cfg['epochs'],
                lr=cfg['learning_rate'],
                epsilon=cfg['epsilon'],
                pgd_steps=cfg['pgd_steps'],
                step_size=cfg['step_size'],
                lambda_adv=cfg['lambda_adv'],
            )
            client_states.append(sd)

        # optional DP + secure aggregation
        client_states = clip_and_add_noise(client_states, clip_norm=cfg['clip_norm'], noise_multiplier=cfg['dp_noise'])
        client_states = secure_aggregate(client_states)

        server.aggregate(client_states)
        print(f"[Round {r}/{rounds}] Aggregated global model.")

    # quick sanity check on random data
    Xtest = torch.rand(1024, cfg['input_dim'])
    ytest = torch.randint(0, cfg['num_classes'], (1024,))
    server.global_model.eval()
    with torch.no_grad():
        logits, _ = server.global_model(Xtest)
        acc = (logits.argmax(1) == ytest).float().mean().item()
        print(f"Synthetic test accuracy (not meaningful): {acc:.3f}")

if __name__ == '__main__':
    main()

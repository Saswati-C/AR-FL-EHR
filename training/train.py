import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from federated.client import load_client_data
from federated.server import federated_averaging
from model.ar_fl_model import ARFLModel
from federated.adversarial_training import fgsm_attack
from privacy.dp_utils import add_dp_to_optimizer


# --------------------------
# Load Config
# --------------------------
def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


# --------------------------
# Local Training (per client)
# --------------------------
def local_train(model, dataloader, criterion, optimizer, device, epsilon, lambda_adv):
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Forward on clean data
        optimizer.zero_grad()
        outputs = model(X)
        loss_clean = criterion(outputs, y)

        # Generate adversarial examples (FGSM)
        X_adv = fgsm_attack(model, X, y, epsilon, criterion)
        outputs_adv = model(X_adv)
        loss_adv = criterion(outputs_adv, y)

        # Mix losses
        loss = (1 - lambda_adv) * loss_clean + lambda_adv * loss_adv
        loss.backward()
        optimizer.step()

    return model.state_dict()


# --------------------------
# Federated Training
# --------------------------
def federated_training(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    input_dim = config["model"]["input_dim"]
    hidden_dims = config["model"]["hidden_dims"]
    num_classes = config["model"]["num_classes"]
    rounds = config["federated"]["rounds"]
    clients = config["federated"]["clients"]
    lr = config["training"]["lr"]
    epsilon = config["adversarial"]["epsilon"]
    lambda_adv = config["adversarial"]["lambda"]

    # Create global model
    global_model = ARFLModel(input_dim, hidden_dims, num_classes).to(device)

    # Load client data
    client_files = [f"data/client_{i+1}.csv" for i in range(clients)]
    client_loaders = [load_client_data(fp, batch_size=32) for fp in client_files]

    for r in range(rounds):
        print(f"\n--- Global Round {r+1} ---")

        client_states = []
        for i, loader in enumerate(client_loaders):
            print(f"Training client {i+1} ...")

            local_model = ARFLModel(input_dim, hidden_dims, num_classes).to(device)
            local_model.load_state_dict(global_model.state_dict())  # start from global

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(local_model.parameters(), lr=lr)

            # Add DP if enabled
            if config["privacy"]["use_dp"]:
                optimizer = add_dp_to_optimizer(optimizer, noise_multiplier=1.0, max_grad_norm=1.0)

            state_dict = local_train(local_model, loader, criterion, optimizer, device, epsilon, lambda_adv)
            client_states.append(state_dict)

        # Aggregate weights (FedAvg)
        global_model.load_state_dict(federated_averaging(client_states))

    torch.save(global_model.state_dict(), "global_model.pth")
    print("✅ Training complete. Model saved as global_model.pth")


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    config_path = "experiments/configs/arfl_config.json"
    config = load_config(config_path)
    federated_training(config)


# Adversarial Robust Federated Learning for Secure Mortality Risk Prediction (AR-FL-EHR)

Official starter implementation scaffolding for the paper:

**Adversarial Robust Federated Learning for Secure Mortality Risk Prediction Using Multi-Institutional EHR**  
*Authors: Saswati Chatterjee, Suneeta Satpathy*

## Overview
This scaffold includes a modular structure for:
- Federated learning (FedAvg) with adversarial training (FGSM/PGD)
- Domain-aware attention for cross-institutional generalization
- Differential privacy hooks and (placeholder) secure aggregation
- Evaluation & visualization utilities

> NOTE: Real EHR datasets are not included. Use links to MIMIC-III / eICU in the README of your final repo and run preprocessing accordingly.

## Quickstart
```bash
pip install -r requirements.txt
python training/train.py --config experiments/configs/arfl_config.json
```


"""Evaluation script.

Replace the synthetic loader with real client/test sets after preprocessing.
"""
import argparse, torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from model.ar_fl_model import ARFLModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=False, help='Path to .pth checkpoint')
    ap.add_argument('--input-dim', type=int, default=256)
    ap.add_argument('--num-classes', type=int, default=2)
    args = ap.parse_args()

    model = ARFLModel(args.input_dim, args.num_classes)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()

    # Synthetic example
    X = torch.rand(1000, args.input_dim)
    y = torch.randint(0, args.num_classes, (1000,))
    with torch.no_grad():
        logits, _ = model(X)
    y_pred = logits.argmax(1).numpy()
    y_true = y.numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Accuracy: {acc:.3f}  F1: {f1:.3f}")

if __name__ == '__main__':
    main()

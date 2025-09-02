
"""Split processed data into multiple clients (institutions).

Usage:
    python -m data_preprocessing.split_clients --input processed.csv --clients 5 --outdir data/clients
"""
import argparse, os, math
import pandas as pd
import numpy as np

def split_data_into_clients(df: pd.DataFrame, num_clients: int):
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    sizes = [len(df)//num_clients]*num_clients
    for i in range(len(df)%num_clients):
        sizes[i] += 1

    chunks, start = [], 0
    for s in sizes:
        chunks.append(df.iloc[start:start+s].reset_index(drop=True))
        start += s
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to processed CSV')
    ap.add_argument('--clients', type=int, default=3, help='Number of clients')
    ap.add_argument('--outdir', default='data/clients', help='Output directory')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)
    splits = split_data_into_clients(df, args.clients)
    for i, d in enumerate(splits):
        path = os.path.join(args.outdir, f'client_{i}.csv')
        d.to_csv(path, index=False)
        print(f'Saved {len(d)} rows to {path}')
    print('Done.')

if __name__ == '__main__':
    main()

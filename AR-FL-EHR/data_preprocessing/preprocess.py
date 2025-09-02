
"""Data preprocessing utilities for EHR features.

- Median imputation for numeric
- One-hot encoding for categorical
- MinMax scaling to [0,1]

Usage:
    python -m data_preprocessing.preprocess --input path/to/raw.csv --output path/to/processed.csv
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # Median imputation for numeric
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())

    # Simple mode imputation for categoricals
    for c in categorical_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])

    # One-hot encode categoricals
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # MinMax scale numeric columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to raw CSV file')
    ap.add_argument('--output', required=True, help='Path to save processed CSV')
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    processed = preprocess_data(df)
    processed.to_csv(args.output, index=False)
    print(f"Saved processed dataset to {args.output} with shape {processed.shape}")

if __name__ == '__main__':
    main()

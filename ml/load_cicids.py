"""
Loader that reads processed parquet files (or a merged parquet) and returns X, y ready for training.

This loader supports MULTICLASS: it expects a saved LabelEncoder (joblib) if you used --save_label_encoder in prepare step.
"""
import os
import glob
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def read_merged_parquet(path):
    return pd.read_parquet(path)

def read_processed_folder(folder_path):
    files = sorted(glob.glob(os.path.join(folder_path, '*.parquet')))
    if not files:
        raise FileNotFoundError('No parquet files found in ' + folder_path)
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    big = pd.concat(dfs, ignore_index=True)
    return big

def get_X_y_from_df(df, label_col='Label', label_encoder_path=None):
    df = df.copy()
    if label_col not in df.columns:
        raise ValueError(f'{label_col} missing from dataframe')
    # encode labels with provided encoder if present
    if label_encoder_path:
        le = joblib.load(label_encoder_path)
        y = le.transform(df[label_col].astype(str))
    else:
        # fallback: simple LabelEncoder
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df[label_col].astype(str))
    # drop label column
    X = df.drop(columns=[label_col])
    # drop any non-numeric columns if present
    X = X.select_dtypes(include=[np.number])
    return X, y

def get_train_test(folder_or_merged_path, label_col='Label', label_encoder_path=None, test_size=0.2, random_state=42):
    if os.path.isdir(folder_or_merged_path):
        df = read_processed_folder(folder_or_merged_path)
    else:
        df = read_merged_parquet(folder_or_merged_path)
    X, y = get_X_y_from_df(df, label_col=label_col, label_encoder_path=label_encoder_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler

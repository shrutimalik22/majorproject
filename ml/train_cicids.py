#!/usr/bin/env python3
"""
Train a multiclass model on processed CICIDS data.
Saves: model.joblib, scaler.joblib
"""
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from ml.load_cicids import get_train_test

DATA_PROCESSED = 'data/processed'  # folder with per-file parquet outputs
MERGED_SAMPLE = None  # or 'data/merged/merged_10pct_stratified.parquet'
LABEL_ENCODER_PATH = 'data/processed/label_encoder.joblib'
MODEL_DIR = 'models'

os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save(merged_path=None, processed_folder=None):
    source = merged_path if merged_path else processed_folder
    if source is None:
        raise ValueError('Please provide merged_path or processed_folder')
    print('Loading data...')
    X_train, X_test, y_train, y_test, scaler = get_train_test(source, label_col='Label', label_encoder_path=LABEL_ENCODER_PATH)
    print('Training RandomForest (multiclass)...')
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    print('Evaluating...')
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    joblib.dump(clf, os.path.join(MODEL_DIR, 'rf_cicids2017_multiclass.joblib'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler_cicids2017.joblib'))
    print('Saved model and scaler to', MODEL_DIR)

if __name__ == '__main__':
    # by default use processed folder
    train_and_save(processed_folder=DATA_PROCESSED)

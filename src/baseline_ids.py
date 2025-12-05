import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


DATA_PATH = "data/cicids_phase1.csv"


def load_data(path=DATA_PATH):
    print("[INFO] Loading combined dataset...")
    df = pd.read_csv(path)
    print("[INFO] Shape:", df.shape)
    print("[INFO] Label counts:")
    print(df["Label"].value_counts())
    return df


def preprocess(df: pd.DataFrame):
    """
    Simple Phase-1 preprocessing:
    - Keep only numeric features
    - Replace inf/-inf with NaN
    - Fill missing numeric values with column mean
    - Map Label: Normal -> 0, Attack -> 1
    """
    print("[INFO] Starting preprocessing...")

    # 1. Map labels to 0/1
    label_map = {"Normal": 0, "Attack": 1}
    df = df[df["Label"].isin(label_map.keys())].copy()
    df["Label"] = df["Label"].map(label_map)

    # 2. Separate features and labels
    y = df["Label"]
    X = df.drop(columns=["Label"])

    # 3. Keep only numeric columns
    X_num = X.select_dtypes(include=["int64", "float64"]).copy()
    print("[INFO] Numeric feature shape:", X_num.shape)

    # 4. Replace inf / -inf with NaN
    X_num = X_num.replace([np.inf, -np.inf], np.nan)

    # 5. Fill NaNs (including former infs) with column mean
    X_num = X_num.fillna(X_num.mean())

    return X_num, y



def train_and_evaluate(X, y):
    print("[INFO] Splitting train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("[INFO] Training RandomForest baseline IDS...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n=== Phase-1 Baseline IDS Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model


def main():
    df = load_data()
    X, y = preprocess(df)
    model = train_and_evaluate(X, y)
    print("\n[INFO] Phase-1 baseline IDS completed successfully.")


if __name__ == "__main__":
    main()

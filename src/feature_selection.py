import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

DATA_PATH = "data/cicids_phase1.csv"
RESULTS_DIR = Path("results")


def load_dataset():
    print("[INFO] Loading dataset for feature selection...")
    df = pd.read_csv(DATA_PATH)

    # Map Label to numeric 0/1
    label_map = {"Normal": 0, "Attack": 1}
    df = df[df["Label"].isin(label_map.keys())].copy()
    df["Label"] = df["Label"].map(label_map)

    # Keep only numeric columns (this will include Label now)
    df_num = df.select_dtypes(include=["int64", "float64"]).copy()

    # Replace inf/-inf with NaN
    df_num = df_num.replace([np.inf, -np.inf], np.nan)

    # Fill NaNs with column means
    df_num = df_num.fillna(df_num.mean())

    print("[INFO] Numeric dataset shape for feature selection:", df_num.shape)
    return df_num


def correlation_selection(df_num: pd.DataFrame):
    print("\n[INFO] Performing correlation-based feature selection...")

    # Correlation with Label (numeric 0/1)
    corr_series = df_num.corr()["Label"].abs().sort_values(ascending=False)

    print("\nTop 20 most correlated features with Label:")
    print(corr_series.head(20))

    RESULTS_DIR.mkdir(exist_ok=True)
    corr_series.to_csv(RESULTS_DIR / "feature_correlation.csv")

    # Keep features with some minimum correlation
    selected_corr_features = corr_series[corr_series > 0.01].index.tolist()

    print(f"\n[INFO] Selected {len(selected_corr_features)} features based on correlation.")

    return selected_corr_features


def random_forest_importance(df_num: pd.DataFrame):
    print("\n[INFO] Calculating feature importances using RandomForest...")

    X = df_num.drop(columns=["Label"])
    y = df_num["Label"]

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    print("\nTop 20 most important features:")
    print(importances.head(20))

    RESULTS_DIR.mkdir(exist_ok=True)
    importances.to_csv(RESULTS_DIR / "random_forest_importance.csv")

    # Keep features with importance above a small threshold
    selected_rf_features = importances[importances > 0.005].index.tolist()
    print(f"\n[INFO] Selected {len(selected_rf_features)} features using RandomForest.")

    return selected_rf_features


def combine_selected_features(corr_features, rf_features):
    final_features = sorted(list(set(corr_features) | set(rf_features)))

    print(f"\n[INFO] Final number of selected features: {len(final_features)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.Series(final_features).to_csv(
        RESULTS_DIR / "selected_features_final.csv", index=False
    )

    return final_features


def main():
    df_num = load_dataset()

    corr_features = correlation_selection(df_num)
    rf_features = random_forest_importance(df_num)

    final_features = combine_selected_features(corr_features, rf_features)

    print("\n[INFO] Feature selection completed!")
    print("Files generated in /results:")
    print("  - feature_correlation.csv")
    print("  - random_forest_importance.csv")
    print("  - selected_features_final.csv")


if __name__ == "__main__":
    main()

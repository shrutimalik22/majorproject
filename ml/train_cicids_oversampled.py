import joblib
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from ml.load_cicids import get_train_test

def train_and_save_oversampled(merged_path, label_encoder_path, out_model, out_scaler):
    # load dataset
    X_train, X_test, y_train, y_test, scaler = get_train_test(
        merged_path, label_col='Label', label_encoder_path=label_encoder_path
    )

    # oversample minority classes
    ros = RandomOverSampler()
    X_res, y_res = ros.fit_resample(X_train, y_train)

    print("\nBefore Oversampling:", X_train.shape, "Class counts:", {i:int(sum(y_train==i)) for i in set(y_train)})
    print("After Oversampling:", X_res.shape, "Class counts:", {i:int(sum(y_res==i)) for i in set(y_res)})

    # train RF
    clf = RandomForestClassifier(n_estimators=300, class_weight=None, random_state=42, n_jobs=-1)
    clf.fit(X_res, y_res)

    # save models
    joblib.dump(clf, out_model)
    joblib.dump(scaler, out_scaler)

    print("\nModel trained with oversampling and saved to:", out_model)

if __name__ == '__main__':
    train_and_save_oversampled(
        merged_path="data/merged/merged_10pct_stratified.parquet",
        label_encoder_path="data/processed/label_encoder.joblib",
        out_model="models/rf_cicids2017_multiclass_oversampled.joblib",
        out_scaler="models/scaler_cicids2017_oversampled.joblib"
    )

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score, top_k_accuracy_score
)
from ml.load_cicids import get_train_test

def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)

def plot_confusion(cm, labels, outpath, normalize=False, figsize=(10,8)):
    if normalize:
        cmn = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)
        mat = cmn
        title = "Confusion matrix (normalized)"
    else:
        mat = cm
        title = "Confusion matrix (counts)"
    plt.figure(figsize=figsize)
    plt.imshow(mat, interpolation='nearest', aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(labels)), labels, fontsize=8)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_roc_pr(y_test, y_score, labels, outdir):
    # y_test: shape (n_samples,) integer encoded
    # y_score: shape (n_samples, n_classes) probability estimates
    n_classes = y_score.shape[1]
    # Binarize true labels
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    # ROC and PR per-class
    roc_auc = {}
    ap_scores = {}
    plt.figure(figsize=(10,8))
    for i in range(n_classes):
        if y_test_bin[:, i].sum() == 0:
            roc_auc[i] = np.nan
            ap_scores[i] = np.nan
            continue
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        r_auc = auc(fpr, tpr)
        roc_auc[i] = r_auc
        plt.plot(fpr, tpr, lw=1, label=f'{labels[i]} (AUC={r_auc:.2f})')
    plt.plot([0,1],[0,1], 'k--', lw=0.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-Rest ROC curves')
    plt.legend(fontsize='small', loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'roc_ovr.png'), dpi=150)
    plt.close()

    # Precision-Recall curves
    plt.figure(figsize=(10,8))
    for i in range(n_classes):
        if y_test_bin[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
        ap_scores[i] = ap
        plt.step(recall, precision, where='post', label=f'{labels[i]} (AP={ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('One-vs-Rest Precision-Recall curves')
    plt.legend(fontsize='small', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'pr_ovr.png'), dpi=150)
    plt.close()

    return roc_auc, ap_scores

def main(args):
    safe_makedirs(args.out_dir)

    # Load label encoder if provided
    label_encoder = None
    if args.label_encoder and os.path.exists(args.label_encoder):
        label_encoder = joblib.load(args.label_encoder)
        labels = list(label_encoder.classes_)
    else:
        labels = None

    # Get train/test. We only need test set here; scaler is loaded inside get_train_test
    if args.merged:
        X_train, X_test, y_train, y_test, scaler = get_train_test(args.merged, label_col='Label', label_encoder_path=args.label_encoder)
    else:
        X_train, X_test, y_train, y_test, scaler = get_train_test(args.processed_folder, label_col='Label', label_encoder_path=args.label_encoder)

    print("Loaded test set:", X_test.shape, "n_classes:", len(np.unique(y_test)))

    # Load model and scaler (scaler from get_train_test is also returned, but load if explicit path given)
    clf = None
    if args.model and os.path.exists(args.model):
        clf = joblib.load(args.model)
    else:
        raise FileNotFoundError("Model file required; pass --model <path>")

    # Predict
    y_pred = clf.predict(X_test)
    # Probabilities for ROC/PR if available
    y_score = None
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)
    elif hasattr(clf, "decision_function"):
        try:
            dec = clf.decision_function(X_test)
            # convert decision function to probabilities via softmax
            from scipy.special import softmax
            y_score = softmax(dec, axis=1)
        except Exception:
            y_score = None

    # Labels for display
    if labels is None:
        # try to build simple labels from unique y values
        uniq = np.unique(np.concatenate([y_train, y_test, y_pred]))
        labels = [str(u) for u in uniq]

    # Overall metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_test, y_pred, average='micro', zero_division=0)

    # Per-class metrics
    prec, rec, f1, sup = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    per_class_df = pd.DataFrame({
        'label_index': np.arange(len(prec)),
        'label': labels[:len(prec)],
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'support': sup
    })

    # Confusion
    cm = confusion_matrix(y_test, y_pred)

    # Save per-class CSV
    per_class_df.to_csv(os.path.join(args.out_dir, 'per_class_evaluation.csv'), index=False)

    # Write text report
    report_txt = []
    report_txt.append("Overall metrics")
    report_txt.append(f"Accuracy: {acc:.4f}")
    report_txt.append(f"Balanced Accuracy: {bal_acc:.4f}")
    report_txt.append(f"Macro Precision/Recall/F1: {p_macro:.4f} / {r_macro:.4f} / {f1_macro:.4f}")
    report_txt.append(f"Micro Precision/Recall/F1: {p_micro:.4f} / {r_micro:.4f} / {f1_micro:.4f}")
    report_txt.append("")
    report_txt.append("Classification report:")
    report_txt.append(classification_report(y_test, y_pred, zero_division=0, target_names=labels[:len(prec)]))
    report_txt.append("")
    report_txt.append("Per-class metrics saved to per_class_evaluation.csv")
    report_txt.append("Confusion matrix shape: " + str(cm.shape))
    with open(os.path.join(args.out_dir, 'eval_report.txt'), 'w') as fh:
        fh.write("\n".join(report_txt))
    print("\n".join(report_txt))

    # Save confusion matrices and plots
    plot_confusion(cm, labels[:cm.shape[0]], os.path.join(args.out_dir, 'confusion_counts.png'), normalize=False)
    plot_confusion(cm, labels[:cm.shape[0]], os.path.join(args.out_dir, 'confusion_normalized.png'), normalize=True)

    # ROC and PR curves (only if y_score present)
    if y_score is not None and y_score.shape[1] == cm.shape[0]:
        roc_auc, ap_scores = plot_roc_pr(y_test, y_score, labels[:cm.shape[0]], args.out_dir)
        # save per-class AUCs
        auc_df = pd.DataFrame({
            'label': labels[:len(roc_auc)],
            'roc_auc': [roc_auc[i] for i in range(len(roc_auc))],
            'avg_precision': [ap_scores.get(i, np.nan) for i in range(len(roc_auc))]
        })
        auc_df.to_csv(os.path.join(args.out_dir, 'per_class_auc.csv'), index=False)
        print("Saved ROC/PR plots and per_class_auc.csv")
    else:
        print("No probability scores available from model - skipping ROC/PR curves")

    # top-k accuracy
    try:
        top3 = top_k_accuracy_score(y_test, clf.predict_proba(X_test), k=3)
        print(f"Top-3 accuracy: {top3:.4f}")
    except Exception:
        print("Top-k accuracy not available (predict_proba probably missing)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--merged', default=None, help='Path to merged parquet (optional)')
    p.add_argument('--processed_folder', default='data/processed', help='Folder with per-file parquet outputs')
    p.add_argument('--model', required=True, help='Path to saved model joblib')
    p.add_argument('--scaler', default=None, help='(not used) path to scaler joblib if you want to load separately')
    p.add_argument('--label_encoder', default='data/processed/label_encoder.joblib', help='Path to saved LabelEncoder')
    p.add_argument('--out_dir', default='models', help='Output folder for reports/images')
    args = p.parse_args()
    main(args)

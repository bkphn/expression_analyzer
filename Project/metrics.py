import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


RED = '\033[91m \x1B[3m'
YELLOW = '\033[93m \x1B[3m'
GREEN = '\033[92m \x1B[3m'
RESET = '\033[0m \x1B[0m'

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                vmin=0,
                vmax=1)

    plt.title('Znormalizowana Macierz Pomyłek (FER-2013)', fontsize=16, pad=20)
    plt.ylabel('Rzeczywista klasa (True Label)', fontsize=14)
    plt.xlabel('Przewidywana klasa (Predicted Label)', fontsize=14)

    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    plt.tight_layout()

    plt.savefig('confusion_matrix_rf.png', dpi=300)
    plt.show()


def print_detailed_metrics(y_true, y_pred, class_names):
    run_metrics(y_true, y_pred, class_names)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)


def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0

def sensitivity_recall(TP, TN, FP, FN):
    return (TP) / (TP + FN) if (TP + FN) > 0 else 0

def presision(TP, TN, FP, FN):
    return (TP) / (TP + FP) if (TP + FP) > 0 else 0

def f1_score(TP, TN, FP, FN):
    return (2*TP) / (2*TP + FP + FN) if (2*TP + FP + FN) > 0 else 0

def specificity(TP, TN, FP, FN):
    return (TN) / (TN + FP) if (TN + FP) > 0 else 0

def run_metrics(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    total_samples = cm.sum()
    metrics = {accuracy : 'Accuracy', sensitivity_recall : 'Sensitivity Recall',
               presision : 'Precision', f1_score : 'F1 Score', specificity : 'Specificity'}

    total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0

    for i, class_name in enumerate(class_names):
        print(f"\nKlasa: {class_name}")

        TP = cm[i, i]
        total_TP += TP
        FP = cm[:, i].sum() - TP
        total_FP += FP
        FN = cm[i, :].sum() - TP
        total_FN += FN
        TN = total_samples - (TP + FP + FN)
        total_TN += TN

        print(f"  True Positives  (TP): {TP}")
        print(f"  False Positives (FP): {FP}")
        print(f"  True Negatives  (TN): {TN}")
        print(f"  False Negatives (FN): {FN}")

        for metric, metric_str in metrics.items():
            score = metric(TP, TN, FP, FN)
            print(f"  {metric_str} = {score*100:.2f}%")

    print(f"\nOgółem:")
    print(f"  True Positives  (TP): {total_TP}")
    print(f"  False Positives (FP): {total_FP}")
    print(f"  True Negatives  (TN): {total_TN}")
    print(f"  False Negatives (FN): {total_FN}")

    for metric, metric_str in metrics.items():
        score = metric(total_TP, total_TN, total_FP, total_FN)
        print(f"  {metric_str} = {score * 100:.2f}%")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


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


def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + FP + FN + TN)

def sensitivity_recall(TP, TN, FP, FN):
    return (TP) / (TP + FN)

def presision(TP, TN, FP, FN):
    return (TP) / (TP + FP)

def f1_score(TP, TN, FP, FN):
    return 0

def specificity(TP, TN, FP, FN):
    return 0

def run_metrics(TP, TN, FP, FN):
    metrics = [accuracy, sensitivity_recall, presision, f1_score, specificity]

    for metric in metrics:
        score = metric(TP, TN, FP, FN)

        print(f"{metric} = {score}")
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


def plot_single_class_cm(TP, TN, FP, FN, class_name):
    cm_2x2 = np.array([[TN, FP],
                       [FN, TP]])

    annot_data = np.array([[f"TN\n{cm_2x2[0, 0]}", f"FP\n{cm_2x2[0, 1]}"],
                           [f"FN\n{cm_2x2[1, 0]}", f"TP\n{cm_2x2[1, 1]}"]])

    plt.figure(figsize=(4, 3))

    sns.heatmap(cm_2x2,
                annot=annot_data,
                fmt='',
                cmap="Blues",
                cbar=False,
                annot_kws={"size": 10},
                xticklabels=['INNA', class_name.upper()],
                yticklabels=['INNA', class_name.upper()])

    plt.title(f'Klasa: {class_name.upper()}', fontsize=12)
    plt.ylabel('Rzeczywista klasa (Actual)', fontsize=10)
    plt.xlabel('Przewidywana klasa (Predicted)', fontsize=10)

    plt.xticks(fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    # plt.savefig(f'cm_{class_name}.png', dpi=300)
    plt.show()

def run_metrics(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    total_samples = cm.sum()
    metrics = {accuracy: 'Accuracy', sensitivity_recall: 'Recall',
               presision: 'Precision', f1_score: 'F1 Score', specificity: 'Specificity'}

    total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0
    table_data = []

    for i, class_name in enumerate(class_names):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = total_samples - (TP + FP + FN)

        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_TN += TN

        row_data = [class_name.upper()]
        for metric, _ in metrics.items():
            row_data.append(metric(TP, TN, FP, FN) * 100)
        table_data.append(row_data)

        plot_single_class_cm(TP, TN, FP, FN, class_name)

    plot_metrics_table(table_data, class_names)

    print("\nOGÓŁEM (Wartości Micro):")
    print(f"  Sumaryczne TP: {total_TP}")
    print(f"  Sumaryczne FP: {total_FP}")
    print(f"  Sumaryczne TN: {total_TN}")
    print(f"  Sumaryczne FN: {total_FN}")

    print("\nGlobalne wskaźniki (Micro-Averaged):")
    for metric, metric_str in metrics.items():
        score = metric(total_TP, total_TN, total_FP, total_FN)
        print(f"  {metric_str:<12} = {score * 100:.2f}%")


def plot_metrics_table(table_data, class_names):
    columns = ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'Specificity']

    cell_text = []
    for row in table_data:
        cell_text.append([f"{val:.2f}%" for val in row[1:]])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_axis_off()

    table = ax.table(
        cellText=cell_text,
        rowLabels=[name.upper() for name in class_names],
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colColours=['#e1e6f0'] * len(columns),
        rowColours=['#e1e6f0'] * len(class_names)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold')

    plt.title('Szczegółowe Metryki Klasyfikacji (One-vs-Rest)', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('metrics_table_rf.png', dpi=300, bbox_inches='tight')
    plt.show()



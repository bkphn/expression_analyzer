import numpy as np
import matplotlib.pyplot as plt
import time
from random_forest import RandomForest


RED = '\033[91m \x1B[3m'
YELLOW = '\033[93m \x1B[3m'
GREEN = '\033[92m \x1B[3m'
RESET = '\033[0m \x1B[0m'


def plot_tree_validation_curve(X_train, y_train, X_test, y_test):
    tree_counts = [1, 5, 10, 20, 30, 50]

    train_scores = []
    test_scores = []

    for n in tree_counts:
        print(f"{YELLOW}Trenowanie Lasu Losowego dla n_estimators = {n}...{RESET}")

        model = RandomForest(n_estimators=n, max_depth=15)
        model.train(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = np.mean(y_train_pred == y_train) * 100
        test_acc = np.mean(y_test_pred == y_test) * 100

        train_scores.append(train_acc)
        test_scores.append(test_acc)

    plt.figure(figsize=(9, 6))

    plt.plot(tree_counts, train_scores, 'o-', color="#1f77b4", label="Zbiór treningowy", linewidth=2.5, markersize=8)
    plt.plot(tree_counts, test_scores, 'o-', color="#ff7f0e", label="Zbiór testowy", linewidth=2.5, markersize=8)

    plt.title("Wpływ liczby drzew na dokładność modelu", fontsize=16, weight='bold', pad=20)
    plt.xlabel("Liczba drzew (n_estimators)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)

    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('validation_curve_trees.png', dpi=300)
    plt.show()

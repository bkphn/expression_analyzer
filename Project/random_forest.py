import random
import numpy as np
from networkx.generators import trees
import concurrent.futures


# ----- CMD COLOR SETTINGS -----
RED = '\033[91m \x1B[3m'
YELLOW = '\033[93m \x1B[3m'
GREEN = '\033[92m \x1B[3m'
RESET = '\033[0m \x1B[0m'

class TreeNode:
    def __init__(self, feature_idx=None, feature_val=None, predicted_class=None):
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.left = None # spełnia warunek
        self.right = None # nie spełnia warunku
        self.predicted_class = predicted_class


class DecisionTree:
    def __init__(self, max_depth=3, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def entropy(self, class_probabilities):
        return sum([-p * np.log2(p) for p in class_probabilities if p > 0])

    def calculate_entropy_for_data(self, labels):
        total_samples = len(labels)
        if total_samples == 0:
            return 0
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / total_samples
        return self.entropy(probabilities)

    def split(self, data, feature_idx, feature_val):
        mask = data[:, feature_idx] < feature_val
        left_split = data[mask]
        right_split = data[~mask]
        return left_split, right_split

    def find_best_split(self, data):
        best_entropy = float('inf')
        best_feature_idx = None
        best_feature_val = None
        best_left = None
        best_right = None

        n_features = data.shape[1] - 1
        max_features = int(np.sqrt(n_features))
        feature_indices = np.random.choice(n_features, size=max_features, replace=False)

        for idx in feature_indices:

            col_min = np.min(data[:, idx])
            col_max = np.max(data[:, idx])

            thresholds = np.linspace(col_min, col_max, num=10)


            for val in thresholds:
                left_split, right_split = self.split(data, idx, val)

                if len(left_split) == 0 or len(right_split) == 0:
                    continue

                n_total = len(data)
                weight_left = len(left_split) / n_total
                weight_right = len(right_split) / n_total

                entropy_left = self.calculate_entropy_for_data(left_split[:, -1])
                entropy_right = self.calculate_entropy_for_data(right_split[:, -1])

                current_entropy = (weight_left * entropy_left) + (weight_right * entropy_right)

                if current_entropy < best_entropy:
                    best_entropy = current_entropy
                    best_feature_idx = idx
                    best_feature_val = val
                    best_left = left_split
                    best_right = right_split

        return best_left, best_right, best_feature_idx, best_feature_val

    def get_majority_class(self, labels):
        values, counts = np.unique(labels, return_counts=True)
        majority_idx = np.argmax(counts)
        return values[majority_idx]

    def create_tree(self, data, current_depth):
        labels = data[:, -1]

        if len(np.unique(labels)) == 1 or current_depth >= self.max_depth or len(labels) <= self.min_samples_leaf:
            return TreeNode(predicted_class=self.get_majority_class(labels))

        left_data, right_data, split_idx, split_val = self.find_best_split(data)

        if left_data is None or right_data is None:
            return TreeNode(predicted_class=self.get_majority_class(labels))

        node = TreeNode(feature_idx=split_idx, feature_val=split_val)
        node.left = self.create_tree(left_data, current_depth + 1)
        node.right = self.create_tree(right_data, current_depth + 1)

        return node

    def train(self, X, y):
        train_data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.tree = self.create_tree(data=train_data, current_depth=0)

    def predict_one_sample(self, X_sample):
        node = self.tree
        while node.predicted_class is None:
            if X_sample[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def predict(self, X_set):
        predictions = [self.predict_one_sample(sample) for sample in X_set]
        return np.array(predictions)


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=3, min_samples_leaf=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees = []

    def bootstrap_sample(self, X_train, y_train):
        N = X_train.shape[0]
        indices = np.random.choice(N, N)
        return X_train[indices], y_train[indices]

    def train_single_tree(self, X, y):
        X_sample, y_sample = self.bootstrap_sample(X, y)

        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf
        )
        tree.train(X_sample, y_sample)
        return tree

    def train(self, X_train, y_train, n_jobs=10):
        i=1
        print(f"{GREEN}\nRozpoczynam trenowanie...{RESET}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(self.train_single_tree, X_train, y_train)
                       for _ in range(self.n_estimators)]

            for future in concurrent.futures.as_completed(futures):
                trained_tree = future.result()
                self.trees.append(trained_tree)
                print(f"{GREEN}Wytrenowano drzewo {i}/{self.n_estimators}.{RESET}")
                i+=1

    def predict(self, X):
        print(f"{GREEN}\nRozpoczynam predykcję...{RESET}")
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        final_predictions = []
        for sample_votes in tree_preds:
            values, counts = np.unique(sample_votes, return_counts=True)
            majority_class = values[np.argmax(counts)]
            final_predictions.append(majority_class)
        print(f"{GREEN}Predykcja zakończona.{RESET}")
        return np.array(final_predictions)
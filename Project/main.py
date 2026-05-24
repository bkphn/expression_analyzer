import os
import pickle
import numpy as np

from preprocessing import preprocess_data
from metrics import plot_confusion_matrix, run_metrics
from random_forest import RandomForest

# ----- MODEL SETTINGS -----
PREPROCESSING = False
TRAIN_MODEL = False

# ----- CMD COLOR SETTINGS -----
RED = '\033[91m \x1B[3m'
YELLOW = '\033[93m \x1B[3m'
GREEN = '\033[92m \x1B[3m'
RESET = '\033[0m \x1B[0m'

# ----- PATHS TO DATA -----
TRAIN_PATH = os.path.join("Data", "train")
TEST_PATH = os.path.join("Data", "test")
MODEL_FILE = os.path.join("Model", "saved_rf_model.pkl")

# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    emotions_map = {"angry": 0, "fear": 1, "happy": 2,
                    "neutral": 3, "sad": 4, "surprise": 5}

    if PREPROCESSING:
        print(f"{GREEN}Rozpoczynam ekstrakcję cech HOG...\n{RESET}")
        train_features, train_labels = preprocess_data(emotions_map, TRAIN_PATH)
        test_features, test_labels = preprocess_data(emotions_map, TEST_PATH)
        print(f"{GREEN}Ekstrakcja cech zakończona pomyślnie.{RESET}")

        print(f"{GREEN}\nGeneruję macierze cech HOG i wektory etykiet...{RESET}")
        X_train = np.array(train_features, dtype=np.float32)
        y_train = np.array(train_labels, dtype=np.float32)
        X_test = np.array(test_features, dtype=np.float32)
        y_test = np.array(test_labels, dtype=np.float32)
        print(f"{GREEN}Pomyślnie stworzono i zapisano pliki.{RESET}")

        print(f"{GREEN}\nZapisuję dane do pliku...{RESET}")
        np.save(os.path.join("Model", "X_train.npy"), X_train)
        np.save(os.path.join("Model", "y_train.npy"), y_train)
        np.save(os.path.join("Model", "X_test.npy"), X_test)
        np.save(os.path.join("Model", "y_test.npy"), y_test)
        print(f"{GREEN}Dane zapisane pomyślnie.{RESET}")
    else:
        print(f"{YELLOW}Wczytuję dane z pliku...{RESET}")
        try:
            X_train = np.load(os.path.join("Model", "X_train.npy"))
            y_train = np.load(os.path.join("Model", "y_train.npy"))
            X_test = np.load(os.path.join("Model", "X_test.npy"))
            y_test = np.load(os.path.join("Model", "y_test.npy"))

        except FileNotFoundError:
            print(f"{RED}Nie znaleziono plików, wygeneruj dane na nowo.{RESET}")
            quit()
        print(f"{YELLOW}Dane wczytano pomyślnie.{RESET}")

    if TRAIN_MODEL:
        print(f"{GREEN}\nRozpoczynam klasyfikację lasem losowym...{RESET}")
        random_forest = RandomForest(100, 15)
        random_forest.train(X_train, y_train)
        print(f"{GREEN}Klasyfikacja zakończona pomyślnie.{RESET}")
        print(f"{GREEN}Zapisuję wytrenowany model do pliku...{RESET}")
        with open(MODEL_FILE, 'wb') as file:
            pickle.dump(random_forest, file)
        print(f"{GREEN}Model zapisany.{RESET}")
    else:
        print(f"{GREEN}Wczytuję gotowy model z dysku...{RESET}")
        with open(MODEL_FILE, 'rb') as file:
            random_forest = pickle.load(file)
        print(f"{GREEN}Model wczytany pomyślnie.{RESET}")

    predictions = random_forest.predict(X_test)

    print(f"{GREEN}\nGeneruję macierz pomyłek...{RESET}")
    emotions_list = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    plot_confusion_matrix(y_test, predictions, emotions_list)
    print(f"{GREEN}Macierz wygenerowana pomyślnie.{RESET}")

    print(f"{GREEN}\nPrzygotowuję metryki modelu...{RESET}")
    run_metrics(y_test, predictions, emotions_list)
    print(f"{GREEN}\nWypisywanie metryk zakończone.{RESET}")

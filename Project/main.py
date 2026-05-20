import os
import numpy as np

from preprocessing import preprocess_data
from metrics import run_metrics

# ----- MODEL SETTINGS -----
PREPROCESSING = False

# ----- CMD COLOR SETTINGS -----
RED = '\033[91m \x1B[3m'
YELLOW = '\033[93m \x1B[3m'
GREEN = '\033[92m \x1B[3m'
RESET = '\033[0m \x1B[0m'

# ----- PATHS TO DATA -----
TRAIN_PATH = os.path.join("Data", "train")
TEST_PATH = os.path.join("Data", "test")

# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    emotions_map = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3,
                    "neutral": 4, "sad": 5, "surprise": 6}

    if PREPROCESSING:
        print(f"{GREEN}Rozpoczynam ekstrakcję cech HOG...\n{RESET}")
        train_features, train_labels = preprocess_data(emotions_map, TRAIN_PATH)
        test_features, test_labels = preprocess_data(emotions_map, TEST_PATH)
        print(f"{GREEN}Ekstrakcja cech zakończona pomyślnie.{RESET}")

        print(f"{GREEN}\nGeneruję macierze cech HOG i wektory etykiet...{RESET}")
        X_train = np.array(train_features)
        y_train = np.array(train_features)
        X_test = np.array(test_features)
        y_test = np.array(test_features)
        print(f"{GREEN}Pomyślnie stworzono i zapisano pliki.{RESET}")

        print(f"{GREEN}\nZapisuję dane do pliku...{RESET}")
        np.save(os.path.join("Model", "X_train.npy"), X_train)
        np.save(os.path.join("Model", "y_train.npy"), y_train)
        np.save(os.path.join("Model", "X_test.npy"), X_test)
        np.save(os.path.join("Model", "y_test.npy"), y_test)
        print(f"{GREEN}Dane zapisane pomyślnie.{RESET}")
    else:
        print(f"{YELLOW}Wczytuję dane z pliku...")
        try:
            X_train = np.load(os.path.join("Model", "X_train.npy"))
            y_train = np.load(os.path.join("Model", "y_train.npy"))
            X_test = np.load(os.path.join("Model", "X_test.npy"))
            y_test = np.load(os.path.join("Model", "y_test.npy"))
        except FileNotFoundError:
            print(f"{RED}Nie znaleziono plików, wygeneruj dane na nowo.{RESET}")
        print(f"{YELLOW}Dane wczytano pomyślnie...")

    #TODO MARTA Tutaj trzeba wywołać drzewa, dane musisz sobie wygenerować
    # i zapiszą się w folderze Model, X to macierz atrybutów, a y to etykiety.

    #TODO MARTA Podliczaj od razu FP, FN, TP, TN będzie łatwiej potem dane z tego wyciągnąć.

    #TODO ADAM Potem tutaj wstawię wywołanie run_metrics()
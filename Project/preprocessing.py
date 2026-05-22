import os
from skimage import exposure
from skimage import io
from skimage.feature import hog
import numpy as np

RED = '\033[91m \x1B[3m'
YELLOW = '\033[93m \x1B[3m'
GREEN = '\033[92m \x1B[3m'
RESET = '\033[0m \x1B[0m'

def extract_features(image):
    return hog(image, orientations=8, pixels_per_cell=(6, 6),
               cells_per_block=(2, 2), transform_sqrt=True, feature_vector=True)

def preprocess_image(image_path):
    image = io.imread(image_path, as_gray=True)
    image = exposure.equalize_adapthist(image, clip_limit=0.03)
    image = image / 255.0

    return image

def preprocess_data(emotions_map, path):
    features = []
    labels = []
    for emotion_name, emotion_label in emotions_map.items():
        emotion_path = os.path.join(path, emotion_name)

        if not os.path.exists(emotion_path):
            print(f"{YELLOW}Nie znaleziono folderu {emotion_path}.{RESET}")
            continue

        for filename in os.listdir(emotion_path):
            if filename.endswith(".jpg"):
                image_path = os.path.join(emotion_path, filename)

                processed_image = preprocess_image(image_path)
                features.append(extract_features(processed_image))
                labels.append(emotion_label)

        print(f"{GREEN}Przetworzono klasę {emotion_name} ({path[5:]}){RESET}")

    return features, labels
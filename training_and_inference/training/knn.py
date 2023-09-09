from sklearn.neighbors import KNeighborsClassifier
from modules.BirdImageDataset import BirdImageDataset

import pandas as pd
import numpy as np

TRAIN_SIZE = 84636 # find splits/train -type f | wc -l
TEST_SIZE = 2626

OUTPUT_PATH = "outputs/knn_score_per_k.csv"
K_VALUES = [2,5,10,20,50,100]

def get_train_test_split():
    x_train = np.empty((TRAIN_SIZE, 224, 224, 3))
    y_train = np.empty((TRAIN_SIZE))

    x_test = np.empty((TEST_SIZE, 224, 224, 3))
    y_test = np.empty((TEST_SIZE))

    for i, (image, label) in enumerate(BirdImageDataset("train", "data/metadata.csv")):
        x_train[i] = image
        y_train[i] = label
        
    for i, (image, label) in enumerate(BirdImageDataset("test", "data/metadata.csv")):
        x_test[i] = image
        y_test[i] = label
        
    return x_train, y_train, x_test, y_test

def get_score_for_k(k, x_train, y_train, x_test, y_test):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(x_train, y_train)
    score = knn_classifier.score(x_test, y_test)
    return score


def main():
    x_train, y_train, x_test, y_test = get_train_test_split()
    k_scores = []
    for k in range(K_VALUES):
        score_cur_k = get_score_for_k(k, x_train, y_train, x_test, y_test)
        k_scores.append(score_cur_k)
        
    df = pd.DataFrame({"k": K_VALUES, "score": k_scores})
    df.to_csv(OUTPUT_PATH, index=False)
    
if __name__ == "__main__":
    main()
from modules.data_prep import get_train_test_split

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import logging

OUTPUT_PATH = "outputs/knn/knn_score_per_k.csv"
K_VALUES = [2, 3, 5, 10, 20, 50, 100, 1000, 5000, 10000, 50000, 10000, 80000]


def get_score_for_k(k, x_train, y_train, x_test, y_test):
    logging.info(f"predicting for k = {k}")
    knn_classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn_classifier.fit(x_train, y_train)
    score = knn_classifier.score(x_test, y_test)
    logging.info(f"score: {score}")
    return score

def main():
    logging.basicConfig(level=logging.INFO)
    start_time = datetime.now()
    logging.info("Starting KNN training")
    logging.info("Starting at:")
    logging.info(start_time)
    x_train, y_train, x_test, y_test = get_train_test_split(path_to_metadata_csv="./metadata/birds.csv", path_to_splits="./splits", resize=(32, 32))
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    logging.info("Finished Loading Dataset")
    
    k_scores = []
    for k in K_VALUES:
        score_cur_k = get_score_for_k(k, x_train, y_train, x_test, y_test)
        k_scores.append(score_cur_k)
    
    end_time = datetime.now()
    total_time = str(end_time - start_time)
    logging.info("Finished training and inference for all k values")
    logging.info("Finished at:")
    logging.info(end_time)
    logging.info(f"Took {total_time}")
        
    df = pd.DataFrame({"k": K_VALUES, "score": k_scores})
    df.to_csv(OUTPUT_PATH, index=False)
    
    plt.plot(K_VALUES, k_scores)
    plt.ylabel("Accuracy")
    plt.xlabel("k")
    plt.title("Accuracy vs k for KNN Bird Image Classifier")
    plt.savefig("outputs/knn/knn_accuracy_vs_k.png")
    plt.clf()
    
if __name__ == "__main__":
    main()
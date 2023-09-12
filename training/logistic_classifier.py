from modules.data_prep import get_train_test_split
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import pandas as pd
import logging

MAX_ITERATIONS = [10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]

def score_for_max_depth(max_iterations, x_train, y_train, x_test, y_test):
    logging.info(f"Training and predicting for max_iterations = {max_iterations}")
    logistic_regression = LogisticRegression(max_iter=max_iterations)
    logistic_regression.fit(x_train, y_train)
    score = logistic_regression.score(x_test, y_test)
    logging.info(f"Accuracy Score: {score}")

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Logistic Regression for Classification")
    logging.info("Starting at:")
    start_time = datetime.now()
    logging.info(start_time)
    x_train, y_train, x_test, y_test = get_train_test_split(path_to_metadata_csv="./metadata/birds.csv", path_to_splits="./splits", resize=(32, 32), fill_zeros=False)
    
    scores_for_max_iterations = []
    
    for max_iterations in MAX_ITERATIONS:
        score = score_for_max_depth(max_iteraions=max_iterations, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        scores_for_max_iterations.append(score)
    
    endtime = datetime.now()
    total_time = str(endtime - start_time)
    
    df = pd.DataFrame({"max_iterations": MAX_ITERATIONS, "score": scores_for_max_iterations})
    df.to_csv("./outputs/logistic_regression/logistic_regression_iterations_vs_score.csv")
    
    logging.info("Finished classification for all max depths")
    logging.info("Finished at:")
    logging.info(endtime)
    logging.info(f"Took {total_time}")

if __name__ == "__main__":
    main()

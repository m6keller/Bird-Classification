from modules.data_prep import get_train_test_split

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

from datetime import datetime
import logging

CRITERION: ["gini", "entropy", "log_loss"] = "gini"
MAX_DEPTHS  = [5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000]

def score_for_max_depth(max_depth, x_train, y_train, x_test, y_test):
    logging.info(f"Predicting for max_depth = {max_depth}")
    decision_tree_classifier = DecisionTreeClassifier(criterion=CRITERION, max_depth=max_depth)
    decision_tree_classifier.fit(x_train, y_train)
    score = decision_tree_classifier.score(x_test, y_test)
    logging.info(f"Score: {score}")
    

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Decision Tree training")
    logging.info("Starting at:")
    start_time = datetime.now()
    logging.info(start_time)
    x_train, y_train, x_test, y_test = get_train_test_split(path_to_metadata_csv="./metadata/birds.csv", path_to_splits="./splits", resize=(32, 32))
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    scores_for_max_depths = []
    
    for max_depth in MAX_DEPTHS:
        score = score_for_max_depth(max_depth, x_train, y_train, x_test, y_test)
        scores_for_max_depths.append(score)
        
    df = pd.DataFrame({"max_depth": MAX_DEPTHS, "score": scores_for_max_depths})
    df.to_csv("./outputs/decision_trees/decision_tree_depth_vs_score.csv")
    
    endtime = datetime.now()
    total_time = str(endtime - start_time)
    logging.info("Finished training and inference for all max depths")
    logging.info("Finished at:")
    logging.info(endtime)
    logging.info(f"Took {total_time}")
    
if __name__ == "__main__":
    main()
    

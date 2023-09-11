from sklearn.tree import DecisionTreeClassifier
from modules.data_prep import get_train_test_split

CRITERION: ["gini", "entropy", "log_loss"] = "gini"

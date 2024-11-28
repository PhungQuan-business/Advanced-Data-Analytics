from pprint import pprint
import pandas as pd
from sklearn.datasets import load_iris

from TraditionalMLAlgorithms import TraditionalMLAlgorithms
from EnsembleMethods import EnsembleMethods
from PreProcessing import pre_processing

data_path = 'Advanced-DA-Task-management.csv'
# data = load_iris()
data = pd.read_csv(data_path)

X, y = pre_processing(data)

# Traditional ML
traditional_ml = TraditionalMLAlgorithms(X, y)

tradditional_algorithms = [
    'logistic_regression', 'decision_tree', 'random_forest',
    'support_vector_machine', 'naive_bayes', 'artificial_neural_network'
]
for algorithm in tradditional_algorithms:
    result = traditional_ml.fit_predict(algorithm_name=algorithm)
    # print("Traditional ML Results:", result)
    pprint(result)


# Ensemble Methods
ensmeble_algorithms = ['bagging', 'boosting']

for algorithm in ensmeble_algorithms:
    ensemble_ml = EnsembleMethods(X, y)
    result = ensemble_ml.fit_predict(algorithm_name=algorithm)
    # print("Ensemble ML Results:", result)
    pprint(result)

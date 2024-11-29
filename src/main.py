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

TRADTIONAL = False
ENSEMBLE = True


if TRADTIONAL:
    tradditional_algorithms = [
        'logistic_regression', 'decision_tree', 'random_forest',
        'support_vector_machine', 'naive_bayes', 'artificial_neural_network'
    ]

    for algorithm in tradditional_algorithms:
        result = traditional_ml.fit_predict(algorithm_name=algorithm)
        # print("Traditional ML Results:", result)
        pprint(result)
        pprint("#"*50)
else:
    print("Traditional method is not enabled")

# Ensemble Methods
if ENSEMBLE:
    ensmeble_algorithms = ['bagging', 'boosting', 'ensemble-voting']

    for algorithm in ensmeble_algorithms:
        ensemble_ml = EnsembleMethods(X, y)
        result = ensemble_ml.fit_predict(algorithm_name=algorithm) #TODO thêm para là mảng gồm các base model muốn thử nghiệm
        # print("Ensemble ML Results:", result)
        pprint(result)
        pprint("#"*50)
else:
    print("Ensemble method is not enabled")


# TODO Thêm feature cho chạy Bagging Boosting cho nhiều base model
# TODO Chưa có feature importance
# TODO có cần bổ sung thêm thuật toán không?

'''
Đã bổ sung cơ chế check algorithm cho single và multiple base model case
format library import 
'''
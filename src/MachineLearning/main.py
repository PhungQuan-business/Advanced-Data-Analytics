from pprint import pprint
import pandas as pd

from Traditional import TraditionalMLAlgorithms
from Ensemble import EnsembleMLAlgorithms
from PreProcessing import pre_processing

data_path = 'src/MachineLearning/Advanced-DA-Task-management.csv'
# data = load_iris()
data = pd.read_csv(data_path)

X, y = pre_processing(data)

# TRADTIONAL = False
# ENSEMBLE = False

# if TRADTIONAL:
#     traditional_ml = TraditionalMLAlgorithms(X, y)
#     tradditional_algorithms = [
#         'logistic_regression', 'decision_tree', 'random_forest',
#         'support_vector_machine', 'naive_bayes', 'artificial_neural_network'
#     ]

#     for algorithm in tradditional_algorithms:
#         result = traditional_ml.fit_predict(algorithm_name=algorithm)
#         # print("Traditional ML Results:", result)
#         pprint(result)
#         pprint("#"*50)
# else:
#     print("Traditional method is not enabled")

all_algorithms = [
    'logistic_regression', 'decision_tree', 'random_forest',
    'support_vector_machine', 'naive_bayes', 'artificial_neural_network',
    'bagging', 'boosting', 'ensemble-voting'
]

if __name__ == "__main__":
    # algorithm = 'logistic_regression'
    algorithm = 'logistic_regression'
    if algorithm in ['bagging', 'boosting', 'ensemble-voting']:
        base_estimator = 'logistic_regression'
        ensemble_ml = EnsembleMLAlgorithms(X, y)
        result = ensemble_ml.fit_predict(
            algorithm_name=algorithm, base_estimator=base_estimator)
        # print("Ensemble ML Results:", result)
        pprint(result)
        pprint("#"*50)
    else:
        traditional_ml = TraditionalMLAlgorithms(X, y)
        result = traditional_ml.fit_predict(
            algorithm_name=algorithm)
        # print("Ensemble ML Results:", result)
        pprint(result)
        pprint("#"*50)
    # if ENSEMBLE:
    #     ensemble_ml = EnsembleMLAlgorithms(X, y)
    #     # ensmeble_algorithms = ['bagging', 'boosting', 'ensemble-voting']
    #     ensmeble_algorithms = ['bagging', 'boosting']

    #     # MLP will make AdaBoost result in error since it doesn't support sample_weight
    #     # SVM make AdaBoost overfitting
    #     estimators = [
    #         'logistic_regression', 'decision_tree', 'random_forest', 'naive_bayes']
    #     for algorithm in ensmeble_algorithms:
    #         for estimator in estimators:
    #             # TODO thêm para là mảng gồm các base model muốn thử nghiệm
    #             result = ensemble_ml.fit_predict(
    #                 algorithm_name=algorithm, base_estimator=estimator)
    #             # print("Ensemble ML Results:", result)
    #             pprint(result)
    #             pprint("#"*50)
    # else:
    #     print("Ensemble method is not enabled")


# TODO Thêm feature cho chạy Bagging Boosting cho nhiều base model
# TODO Chưa có feature importance
# TODO có cần bổ sung thêm thuật toán không?

'''
Đã bổ sung cơ chế check algorithm cho single và multiple base model case
format library import

bổ sung danh sách default params cho traditional model, có thể chạy dùng default params hoặc custom

ERROR:
ENSEMBLE chạy ổn
Lỗi khi chạy Traditional, sau khi thêm multi_class='ovr' đoạn lấy roc-auc thì bị lỗi index out of range
'''

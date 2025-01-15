import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from MLAlgorithmBase import MLAlgorithmBase
from MLAlgorithmBase import ModelParameters


class TraditionalModelParams(ModelParameters):
    def __init__(self):
        super().__init__()
        self.default_params = {
            "logistic_regression": {
                "penalty": "l2",
                "C": 1.0,
                "max_iter": 1000,
                "random_state": 42,
            },
            "decision_tree": {
                "criterion": "gini",
                "max_depth": None,
                "random_state": 42,
            },
            "random_forest": {
                "n_estimators": 100,
                "criterion": "gini",
                "max_depth": None,
                "random_state": 42,
            },
            "support_vector_machine": {
                "C": 1.0,
                "kernel": "rbf",
                "degree": 3,
                "probability": True,
                "decision_function_shape": 'ovr',
                "random_state": 42,
            },
            "naive_bayes": {},  # Naive Bayes often doesn't require complex parameters
            "artificial_neural_network": {
                "hidden_layer_sizes": (100,),
                "activation": "relu",
                "solver": "adam",
                "max_iter": 1000,
                "random_state": 42,
            },
        }


class TraditionalMLAlgorithms(MLAlgorithmBase):
    def __init__(self, X, y, test_size=0.3, random_state=42):
        super().__init__(X, y, test_size, random_state)

    def fit_predict(self, algorithm_name, base_estimator=None):
        """
        Select and apply traditional ML algorithms.

        Parameters:
        -----------
        algorithm_name : str
            Name of the algorithm to apply

        Returns:
        --------
        dict
            Results of the model including predictions, accuracy, and classification report
        """
        supported_algorithms = [
            'logistic_regression', 'decision_tree', 'random_forest',
            'support_vector_machine', 'naive_bayes', 'artificial_neural_network'
        ]
        algorithm = self._validate_input(algorithm_name, supported_algorithms)
        get_trad_param = TraditionalModelParams()
        params = get_trad_param.get_params(
            algorithm_name=algorithm, custom_params=base_estimator)

        if algorithm == 'logistic_regression':
            model = LogisticRegression(**params)
        elif algorithm == 'decision_tree':
            model = DecisionTreeClassifier(**params)
        elif algorithm == 'random_forest':
            model = RandomForestClassifier(**params)
        elif algorithm == 'support_vector_machine':
            model = SVC(**params)
        elif algorithm == 'naive_bayes':
            model = GaussianNB(**params)
        elif algorithm == 'artificial_neural_network':
            model = MLPClassifier(**params)

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred, multi_class='ovo')
        report = classification_report(self.y_test, y_pred, zero_division=1)
        return {
            'model': model,
            'ROC AUC Score': roc_auc,
            'Accuracy': accuracy,
            'Classification Report': report
        }


if __name__ == '__main__':
    import pprint as pprint
    from sklearn.datasets import load_iris

    data = load_iris()
    X, y = data.data, data.target

    traditional_ml = TraditionalMLAlgorithms(X, y)
    tradditional_algorithms = [
        'logistic_regression', 'decision_tree', 'random_forest',
        'support_vector_machine', 'naive_bayes', 'artificial_neural_network'
    ]

    for algorithm in tradditional_algorithms:
        result = traditional_ml.fit_predict(algorithm_name=algorithm)
        # print("Traditional ML Results:", result)
        pprint(result)
        pprint("#"*50)


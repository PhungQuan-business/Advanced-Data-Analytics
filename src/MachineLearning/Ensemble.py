import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from MLAlgorithmBase import MLAlgorithmBase
from MLAlgorithmBase import ModelParameters
# # Optional imports for additional algorithms
# try:
#     import xgboost as xgb
#     XGBOOST_AVAILABLE = True
# except ImportError:
#     XGBOOST_AVAILABLE = False

# try:
#     import lightgbm as lgb
#     LIGHTGBM_AVAILABLE = True
# except ImportError:
#     LIGHTGBM_AVAILABLE = False

MODEL_CLASSES = {
    'logistic_regression': LogisticRegression,
    'decision_tree': DecisionTreeClassifier,
    'random_forest': RandomForestClassifier,
    'support_vector_machine': SVC,
    'naive_bayes': GaussianNB,
    'artificial_neural_network': MLPClassifier
}


class EnsembleModelParams(ModelParameters):
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


class EnsembleMLAlgorithms(MLAlgorithmBase):
    def __init__(self, X, y, test_size=0.3, random_state=42):
        super().__init__(X, y, test_size, random_state)

    def fit_predict(self, algorithm_name, base_estimator=None):
        """
        Select and apply ensemble methods (Bagging and Boosting).

        Parameters:
        -----------
        algorithm_name : str
            Name of the algorithm to apply

        Returns:
        --------
        dict
            Results of the model including predictions, accuracy, and classification report
        """

        supported_algorithms = ['bagging', 'boosting', 'ensemble-voting']
        supported_base_algorithms = [
            'logistic_regression', 'decision_tree', 'random_forest',
            'support_vector_machine', 'naive_bayes', 'artificial_neural_network'
        ]

        algorithm = self._validate_input(algorithm_name, supported_algorithms)
        if base_estimator == None:
            base_model_name = 'decision_tree'
            get_ensem_param = EnsembleModelParams()
            get_base_model_params = get_ensem_param.get_params(
                algorithm_name=base_model_name, custom_params=None)
            base_model = base_model_class(**get_base_model_params)
        else:
            base_model_name = self._validate_input(
                base_estimator, supported_base_algorithms)

            # TODO add iteration through list of alogirthms
            # TODO modify the result printing

            base_model_name_lower = base_model_name.lower().strip()
            if base_model_name_lower not in MODEL_CLASSES:
                raise ValueError(
                    f"Unsupported base model: {base_model_name}. Supported models are: {list(MODEL_CLASSES.keys())}"
                )

            base_model_class = MODEL_CLASSES[base_model_name_lower]
            get_ensem_param = EnsembleModelParams()
            get_base_model_params = get_ensem_param.get_params(
                algorithm_name=base_model_name, custom_params=None)
            base_model = base_model_class(**get_base_model_params)

        if algorithm == 'bagging':
            model = BaggingClassifier(
                estimator=base_model,
                n_estimators=10,
                random_state=42
            )
        # Use SVM with AdaBoost result in overfitting
        # use MLPClassifier with AdaBoost result in error since it doesn't support sample_weight, not a bug
        elif algorithm == 'boosting':
            model = AdaBoostClassifier(
                algorithm='SAMME',
                estimator=base_model,
                n_estimators=50,
                random_state=42
            )
        elif algorithm == 'ensemble-voting':
            # Voting Classifier with multiple base models
            base_models = [
                ('lr', DecisionTreeClassifier(random_state=42)),
                ('rf', RandomForestClassifier(random_state=42)),
                ('svm', MLPClassifier(hidden_layer_sizes=(
                    50, 25), max_iter=500, random_state=42))
            ]

            model = VotingClassifier(
                estimators=base_models,
                voting='soft'  # Use soft voting with probability predictions
            )
        else:
            raise ValueError(f"Algorithm {algorithm_name} is not implemented")

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        # Result
        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, zero_division=1)
        return {
            'model': model,
            # y_pred,
            'ROC AUC Score': roc_auc,
            'Accuracy': accuracy,
            'Classification Report': report
        }


if __name__ == '__main__':
    import pprint as pprint
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data.data, data.target
    ensemble_ml = EnsembleMLAlgorithms(X, y)
    ensmeble_algorithms = ['bagging', 'boosting']

    # MLP will make AdaBoost result in error since it doesn't support sample_weight
    # SVM make AdaBoost overfitting
    estimators = [
        'logistic_regression', 'decision_tree', 'random_forest', 'naive_bayes']
    for algorithm in ensmeble_algorithms:
        for estimator in estimators:
            # TODO thêm para là mảng gồm các base model muốn thử nghiệm
            result = ensemble_ml.fit_predict(
                algorithm_name=algorithm, base_estimator=estimator)
            # print("Ensemble ML Results:", result)
            pprint(result)
            pprint("#"*50)
'''
khi main cung cấp tên thuật toán (vòng lặp for)
code sẽ check xem thuật toán có support không
nếu có thì khởi tạo class của function đó cùng với params của Class
'''

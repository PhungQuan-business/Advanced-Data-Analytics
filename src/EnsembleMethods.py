import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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


class EnsembleMethods(MLAlgorithmBase):
    def fit_predict(self, algorithm_name, base_estimator):
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
        base_models = self._validate_input(
            base_estimator, supported_base_algorithms)

        # TODO add iteration through list of alogirthms
        # TODO modify the result printing

        '''
        Thêm vòng for để loop qua các base model
        '''

        if algorithm == 'bagging':
            model = BaggingClassifier(
                estimator=DecisionTreeClassifier(),
                n_estimators=10,
                random_state=42
            )
        elif algorithm == 'boosting':
            model = AdaBoostClassifier(
                algorithm='SAMME',
                estimator=DecisionTreeClassifier(max_depth=1),
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

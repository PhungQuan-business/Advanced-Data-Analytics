import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from pgmpy.models import BayesianNetwork
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
    def fit_predict(self, algorithm_name):
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
        supported_algorithms = ['bagging', 'boosting']
        algorithm = self._validate_input(algorithm_name, supported_algorithms)

        if algorithm == 'bagging':
            model = BaggingClassifier(
                estimator=DecisionTreeClassifier(),
                n_estimators=10,
                random_state=42
            )
        elif algorithm == 'boosting':
            model = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=50,
                random_state=42
            )

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, zero_division=1)

        # return {
        #     'model': model,
        #     'predictions': y_pred,
        #     'accuracy': accuracy,
        #     'classification_report': report
        # }
        return (
            model,
            y_pred,
            accuracy,
            report
        )

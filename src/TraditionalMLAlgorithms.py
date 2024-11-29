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

# Kế thừa 1 class với các function _validate_input(), fit_preidct(),


class TraditionalMLAlgorithms(MLAlgorithmBase):
    def fit_predict(self, algorithm_name):
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

        if algorithm == 'logistic_regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif algorithm == 'decision_tree':
            model = DecisionTreeClassifier(random_state=42)
        elif algorithm == 'random_forest':
            model = RandomForestClassifier(random_state=42)
        elif algorithm == 'support_vector_machine':
            model = SVC(random_state=42, decision_function_shape='ovr')
        elif algorithm == 'naive_bayes':
            model = GaussianNB()
        elif algorithm == 'artificial_neural_network':
            model = MLPClassifier(hidden_layer_sizes=(
                50, 25), max_iter=500, random_state=42)

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

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

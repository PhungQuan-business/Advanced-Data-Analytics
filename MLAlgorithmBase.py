import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MLAlgorithmBase:
    def __init__(self, X, y, test_size=0.3, random_state=42):
        """
        Initialize the ML Algorithm Base Class

        Parameters:
        -----------
        X : numpy array or pandas DataFrame
            Input features
        y : numpy array or pandas Series
            Target labels
        test_size : float, optional (default=0.2)
            Proportion of the dataset to include in the test split
        random_state : int, optional (default=42)
            Controls the shuffling applied to the data before splitting
        """
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, y, test_size=test_size, random_state=random_state
        )

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

    def _validate_input(self, algorithm_name, supported_algorithms):
        """
        Validate the input algorithm name against supported algorithms.

        Parameters:
        -----------
        algorithm_name : str
            Name of the algorithm to validate
        supported_algorithms : list
            List of supported algorithm names

        Returns:
        --------
        str
            Standardized algorithm name
        """
        algorithm_name_lower = algorithm_name.lower().strip()
        if algorithm_name_lower not in supported_algorithms:
            raise ValueError(
                f"Unsupported algorithm: {algorithm_name}. Supported algorithms are: {supported_algorithms}"
            )
        return algorithm_name_lower

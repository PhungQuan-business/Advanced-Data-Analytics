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

    def _validate_input(self, algorithm_names, supported_algorithms):
        """
        Validates the input algorithm name(s) against supported algorithms.

        Parameters:
        -----------
        algorithm_names : str or list
            Name(s) of the algorithm(s) to validate.
        supported_algorithms : list
            List of supported algorithm names.

        Returns:
        --------
        str or list
            Standardized algorithm name(s).

        Raises:
        -------
        ValueError
            If any algorithm name is unsupported or invalid.
        """
        if isinstance(algorithm_names, str):  # Single algorithm case
            algorithm_name_lower = algorithm_names.lower().strip()
            if algorithm_name_lower not in supported_algorithms:
                raise ValueError(
                    f"Unsupported algorithm: {algorithm_names}. "
                    f"Supported algorithms are: {supported_algorithms}"
                )
            return algorithm_name_lower

        elif isinstance(algorithm_names, list):  # List of algorithms case
            unsupported = [
                algo for algo in algorithm_names if algo.lower().strip() not in supported_algorithms
            ]
            if unsupported:
                raise ValueError(
                    f"Unsupported algorithms detected: {unsupported}. "
                    f"Supported algorithms are: {supported_algorithms}"
                )
            return [algo.lower().strip() for algo in algorithm_names]

        else:  # Invalid type case
            raise ValueError(
                f"Invalid input type: {type(algorithm_names).__name__}. "
                f"Expected a string or a list of strings."
            )

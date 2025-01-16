'''
import file
read file
choose model
'''
import streamlit as st
import pandas as pd
from pprint import pprint
from MachineLearning.Traditional import TraditionalMLAlgorithms
from MachineLearning.Ensemble import EnsembleMLAlgorithms
from MachineLearning.PreProcessing import pre_processing


def main():
    st.title("Machine Learning Algorithm Demo")
    st.sidebar.title("Configuration")

    # Upload dataset
    data_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

    if data_file is not None:
        data = pd.read_csv(data_file)
        st.write("### Dataset Preview")
        st.write(data.head())

        # Algorithm selection
        algorithms = [
            'logistic_regression', 'decision_tree', 'random_forest',
            'support_vector_machine', 'naive_bayes', 'artificial_neural_network',
            'bagging', 'boosting', 'ensemble-voting'
        ]
        algorithm = st.sidebar.selectbox("Choose Algorithm", algorithms)

        # Base estimator for ensemble methods
        base_estimator = None
        if algorithm in ['bagging', 'boosting', 'ensemble-voting']:
            X, y = pre_processing(data)
            base_estimators = [
                'logistic_regression', 'decision_tree', 'random_forest',
                'support_vector_machine', 'naive_bayes', 'artificial_neural_network'
            ]
            base_estimator = st.sidebar.selectbox(
                "Choose Base Estimator", base_estimators)

        # Run selected algorithm
        try:
            if st.sidebar.button("Run Algorithm"):
                st.write(f"### Running {algorithm}")

                if algorithm in ['bagging', 'boosting', 'ensemble-voting']:
                    ensemble_ml = EnsembleMLAlgorithms(X, y)
                    result = ensemble_ml.fit_predict(
                        algorithm_name=algorithm, base_estimator=base_estimator)
                else:
                    traditional_ml = TraditionalMLAlgorithms(X, y)
                    result = traditional_ml.fit_predict(
                        algorithm_name=algorithm)

                # Display results
                st.write("### Results")
                st.json(result)
        except Exception as e:
            st.error(f"{str(e)}")


if __name__ == "__main__":
    main()

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
from DeepLearning.DLTabularModels import TabularModels

agorithm_type = ['Tradition', 'Ensemble', 'Deep Learning']

tradition_algorithm = ['logistic_regression', 'decision_tree', 
                       'random_forest','support_vector_machine', 
                       'naive_bayes', 'artificial_neural_network'
                       ]

ensemble_agorithm = ['bagging', 'boosting', 'ensemble-voting']

deep_learning_algorithm = ['NODE', 'TabNet', 'AutoInt', 'TabTransformer', 'GATE', 'GANDAF', 'DANETs']

base_estimators = [
    'logistic_regression', 'decision_tree', 'random_forest',
    'support_vector_machine', 'naive_bayes', 'artificial_neural_network'
    ]

loss_functions = [
    'DiceLoss','DiceBCELoss',
    'IoULoss','FocalLoss',
    'TverskyLoss','FocalTverskyLoss'
    ]

def main():
    st.title("Financial Restatement engine")
    st.sidebar.title("Configuration")

    # Upload dataset
    data_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

    if data_file is not None:
        data = pd.read_csv(data_file)
        st.write("### Dataset Preview")
        st.write(data.head())

        # Algorithm selection
        type = st.sidebar.selectbox("Choose Algorithm type", agorithm_type)
        if type == 'Tradition':
            algorithm = st.sidebar.selectbox("Choose Algorithm", tradition_algorithm)
            
        elif type == 'Ensemble':
            algorithm = st.sidebar.selectbox("Choose Algorithm", ensemble_agorithm)
            # Base estimator for ensemble methods
            base_estimator = None
            base_estimator = st.sidebar.selectbox(
                "Choose Base Estimator", base_estimators)
        elif type == 'Deep Learning':
            algorithm = st.sidebar.selectbox("Choose Algorithm", deep_learning_algorithm)
            loss_function = st.sidebar.selectbox(
                "Choose Loss Function", loss_functions)

            
        # Run selected algorithm
        try:
            if st.sidebar.button("Run Algorithm"):
                st.write(f"### Running {algorithm}")

                if algorithm in ['bagging', 'boosting', 'ensemble-voting']:
                    X, y = pre_processing(data)
                    ensemble_ml = EnsembleMLAlgorithms(X, y)
                    result = ensemble_ml.fit_predict(
                        algorithm_name=algorithm, base_estimator=base_estimator)
                elif algorithm in ['NODE', 'TabNet', 'AutoInt', 'TabTransformer', 'GATE', 'GANDAF', 'DANETs']:
                    from sklearn.model_selection import train_test_split
                    processed_data = pre_processing(data, pytorch=True)
                    cat_col_names = []
                    target_col = "R/Not-R"
                    num_col_names = processed_data.select_dtypes(include=['float64'])
                    tabular_model = TabularModels(target_col, num_col_names, 
                                                  cat_col_names, 1, algorithm)
                    
                    train, test = train_test_split(processed_data, random_state=42, test_size=0.2)
                    train, val = train_test_split(train, random_state=42, test_size=0.2)
                
                    result = tabular_model.fit_predict(train, val, test, algorithm, loss_function)
                else:
                    X, y = pre_processing(data)
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

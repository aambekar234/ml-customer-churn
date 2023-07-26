'''
Author: A. Ambekar
Date: 06/29/2023
'''

import os
from churn_library import perform_eda
from churn_library import encoder_helper
from churn_library import perform_feature_engineering
from churn_library import train_models
from churn_library import import_data
import logging.config
import pytest
logging.config.fileConfig("log_config.ini")
logger = logging.getLogger()

@pytest.fixture
def load_data():
    df = None
    try:
        df = import_data("./data/data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    return df

@pytest.fixture
def split_data(load_data):
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    data = encoder_helper(load_data, cat_columns, 'Churn')
    return perform_feature_engineering(data, 'Churn')

def test_import(load_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        assert load_data.shape[0] > 0
        assert load_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

def test_eda(load_data):
    '''
    test perform eda function
    '''

    perform_eda(load_data)
    figures_path = "images/eda/"
    image_list = ['churn_distribution.png', 'heatmap.png', 
    'marital_status_distribution.png', 'total_transaction_distribution.png', 
    'customer_age_distribution.png']

    try:
        for image in image_list:
            assert os.path.exists(os.path.join(figures_path,image)) == True
        logger.info("Successfully completed test eda!")
    except AssertionError as err:
        logger.warning("Failed test eda!")
        raise err

def test_encoder_helper(load_data):
    '''
    test encoder helper
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    data = encoder_helper(load_data, cat_columns, 'Churn')

    try:
        for column in cat_columns:
            column_name = f"{column}_Churn"
            assert column_name in data.columns
        logger.info("Successfully completed test encoder_helper!")
    except AssertionError as err:
        logger.error("Failed test encoder_helper!")
        raise err

    return data

def test_perform_feature_engineering(load_data):
    '''
    test perform_feature_engineering
    '''

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    data = encoder_helper(load_data, cat_columns, 'Churn')
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        data, 'Churn')

    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logger.info("Successfully completed test perform_feature_engineering!")
    except AssertionError as err:
        logger.error("Failed test perform_feature_engineering!")
        raise err


def test_train_models(split_data):
    '''
    test train_models
    '''

    X_train, X_test, y_train, y_test = split_data
    train_models(X_train, X_test, y_train, y_test)
   
    artifacts_list = ["models/logistic_model.pkl", 
    "models/rfc_model.pkl",
    "images/results/roc_curve_result.png",
    "images/results/test_logistic_regression.png",
    "images/results/train_logistic_regression.png",
    "images/results/test_random_forest.png",
    "images/results/train_random_forest.png",
    "images/results/feature_importance.png"]

    try:
        for artifact in artifacts_list:
            assert os.path.exists(artifact)
        logger.info("Successfully completed test train_models!")
    except AssertionError as err:
        logger.warning("Failed test train_models!")
        raise err

if __name__ == "__main__":
    pytest.main(["./churn_script_logging_and_tests.py"])

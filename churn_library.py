'''
Author: A. Ambekar
Date: 06/29/2023
'''

# import libraries
import os
import logging.config
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

os.makedirs("./logs/")
logging.config.fileConfig("log_config.ini")
logger = logging.getLogger()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    with open(pth) as fp:
        df = pd.read_csv(fp)
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    figures_path = "images/eda/"
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    # churn distrinution
    plt.figure(figsize=(20, 10))
    fig, ax = plt.subplots()
    df.hist('Churn', ax=ax)
    fig.savefig(os.path.join(figures_path, 'churn_distribution.png'))
    plt.close()

    # customer age distribution
    plt.figure(figsize=(20, 10))
    fig, ax = plt.subplots()
    df.hist('Customer_Age', ax=ax)
    fig.savefig(os.path.join(figures_path, 'customer_age_distribution.png'))
    plt.close()

    # heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df.corr(
            numeric_only=True),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.title("Heatmap")
    plt.savefig(os.path.join(figures_path, 'heatmap.png'))
    plt.close()

    # martial status dstribution
    plt.figure(figsize=(20, 10))
    plt.title("Marital status distribution.")
    fig, ax = plt.subplots()
    fig = df.Marital_Status.value_counts(
        'normalize').plot(kind='bar').get_figure()
    fig.savefig(os.path.join(figures_path, 'marital_status_distribution.png'))
    plt.close()

    # total transaction distribution
    plt.figure(figsize=(20, 10))
    plt.title("Total transaction distribution.")
    fig, ax = plt.subplots()
    fig = sns.histplot(df['Total_Trans_Ct'],
                       stat='density', kde=True).get_figure()
    fig.savefig(os.path.join(
        figures_path, "total_transaction_distribution.png"))
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the
    notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
            be used for naming variables or index y column]
    output:
            df: pandas dataframe with new columns for
    '''

    for category in category_lst:
        lst = []
        groups = df.groupby(category).mean(numeric_only=True)[response]
        for val in df[category]:
            lst.append(groups.loc[val])

        df[f'{category}_{response}'] = lst

    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could
              be used for naming variables or index y column]
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df['Churn']
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X = pd.DataFrame()
    data = encoder_helper(df, cat_columns, response)
    X[keep_cols] = data[keep_cols]

    # train test split
    return train_test_split(X, y, test_size=0.3, random_state=42)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds,
                                filename):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions
            y_test_preds: testing predictions
            filename: filename string for saving the report

    output:
             None
    '''

    # generate reports and save
    target_names = ['0', '1']
    save_classification_report(
        classification_report(
            y_train,
            y_train_preds,
            target_names=target_names),
        f"train_{filename}")
    save_classification_report(classification_report(
        y_test, y_test_preds, target_names=target_names), f"test_{filename}")


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth, dpi=300)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Grid search, Random forest

    logger.info("Checking necessary directories exist for saving artifacts.")
    if not os.path.exists("./models/"):
        os.makedirs("./models")

    if not os.path.exists("./images/results"):
        os.makedirs("./images/results")

    # tain logistic regression and save model
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    joblib.dump(lrc, './models/logistic_model.pkl')

    # generate classification report images for LR
    classification_report_image(
        y_train, y_test, y_train_preds_lr, y_test_preds_lr,
        "logistic_regression.png")

    # train Random forest with Grid search and save the best model
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')

    # generate classification report images for RF
    classification_report_image(
        y_train, y_test, y_train_preds_rf, y_test_preds_rf,
        "random_forest.png")

    # plot roc curve and save
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_,
                              X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("images/results/roc_curve_result.png", dpi=300)
    plt.close()

    # plot feature importance and save
    save_feature_importance_graph(X_train)


def save_feature_importance_graph(df):
    '''
    plots feature importance and saves the figure
    input:
              df: pandas data frame used for trainig
    output:
              None
    '''
    # Sort feature importances in descending order
    rfc = joblib.load("./models/rfc_model.pkl")
    feature_importances = rfc.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [df.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(30, 20))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(df.shape[1]), feature_importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(df.shape[1]), names, rotation=90)
    plt.savefig("images/results/feature_importance.png", dpi=300)
    plt.close()


def save_classification_report(report, filename):
    '''
    Convert classification report text into an image
    input:
            report: report generated by sklearn classification_report function
            filename: name of the output image file
    output:
            None
    '''

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    lines = report.split("\n")
    table_data = []
    for line in lines[2:-5]:  # Exclude the first two lines and last five lines
        row_data = line.split()
        table_data.append(row_data)

    ax.table(cellText=table_data,
             colLabels=["class", "precision", "recall", "f1-score", "support"],
             cellLoc="center", loc="center")
    plt.tight_layout()

    # Save the image as a PNG file
    figure_path = "images/results/"
    plt.savefig(os.path.join(figure_path, filename), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    logger.info("Reading data file...")
    data = import_data("./data/data.csv")

    logger.info("Performing EDA...")
    perform_eda(data)

    logger.info("Splitting data into Train & Test.")
    X_train, X_Test, y_train, y_test = perform_feature_engineering(
        data, 'Churn')

    logger.info("Train models and save the artifacts.")
    train_models(X_train, X_Test, y_train, y_test)

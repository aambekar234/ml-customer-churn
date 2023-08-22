# Predict Customer Churn
<br>

## Project Description

In this project, you will find a way to implement, identifying credit card customers that are most likely to churn. The project includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).

This project will showcase necessary skills for testing, logging, and best coding practices. It will also introduce you to a problem data scientists across companies face all the time. How do we identify (and later intervene with) customers who are likely to churn?

Data is pulled from this [Kaggle Source!](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code)

## How to run? 

- Install conda/minicoda. [miniconda](https://docs.conda.io/en/latest/miniconda.html#installing)
- Create conda environement by below command and activate the environment
    ```
    conda env create -f environment.yml --force
    conda activate ml-customer-churn
    ```
- Run experiment file
    ```
    python churn_library.py
    ```
- Run unit tests
    ```
    python churn_script_logging_and_tests.py
    ```

## Files in the Repo

- data
    - bank_data.csv (*data file*)
- churn_library.py (*source code*)
- churn_notebook.ipynb (*Ipython notebook of the experiment*)
- churn_script_logging_and_tests.py (*unit tests*)
- environment.yml (*conda environment file*)
- log_config.ini (*application logs configuration file*)
- README.md 
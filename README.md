
**FraudDetection**

    Description
**FraudDetection** (Fraud Detection)is Package built in Python for fraud transactions in a banking system using supervised machine learning. It pulls data from the `Kaggle databaset <https://www.kaggle.com/rupakroy/online-payments-fraud-detection-dataset>`_ and offers a *simple* and *intuitive* solution.

Badges:

[![GitHub](https://img.shields.io/badge/GitHub-ArunKhare/FraudDetection.git-blue.svg)](https://github.com/ArunKhare/FraudDetection.git)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![documentation](https://github.com/ArunKhare/FraudDetection/actions/workflows/update-gh-pages.yml/badge.svg?branch=docs-master)](https://github.com/ArunKhare/FraudDetection/actions/workflows/update-gh-pages.yml)

Table of Contents:

Create a table of contents with links to important sections within your README.

Installation:

    Create a Conda environment in root directory of the project:
        - This will install python 3.10 all dependencies from *'requirements_dev.txt'* fro the project:

        bash init_setup.sh
    
    Instruction to run the project:
        conda activate ./FraudDetection
    Save the environment:
        conda env export --file conda.yaml
    
Usage:
    for running the project for training and predictions from root directory for the project:
        usage senarios: 
            1 from console 
                - Training 
                    python apps\app.py
                - Prediction
                    python src\fraudDetection\components\prediction\prediction_service.py
            2 Stremalit app
                python apps/tranningapp.py
                    - run 'streamlit run <path>/trainingapp.py' 
                    - This will run the Streamlit sever in the background
                        You can now view your Streamlit app in your browser.
                        Local URL: http://localhost:8501
                        Network URL: http://192.168.99.138:8501             

                     A friendly UI will guide you though its varous usage
                            -for training and prediction

        for screen shots  and  video <>
    Alternatively if are editing the code:
        python dvc init
        python dvc repro
    running mlflowUI:
        run mlflow ui
        - This runs the Mlflow UI server in the background
        - Click on link http://127.0.0.1:5000

    Environment Variable:
    use *<root_dir>/.env*
        *MLFLOW_TRACKING_URI=sqlite:///mlruns.db*
        
    Kaggle Authentication:
        - Download the kaggle authentication from Kaggle setting as kaggle.json file
        - Place the file in *<root>/.kaggle*
    
    For testing code:
        FraudDetection Project is configured with pytest

    configure your project for their specific needs using Config files:
        - tox.ini
        - pyptoject.toml
        
Contributing:
    1. Links and Details:
        - https://github.com/ArunKhare/FraudDetection/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22
        - Fork, install the project as mentiond in Installation section, Test the code using pytest. Create a pull request
            Contributors should provide a clear title and description for their pull request. This description should include details about the changes made, the purpose of the changes, and any context that reviewers may need.
    2. Codind Standard
        follow the [Black](https://github.com/psf/black) code style for this project. Black is an opinionated code formatter that ensures consistent formatting across the codebase.
        To ensure code consistency and readability, we recommend running Black before submitting any code changes. 
        If you haven't installed Black yet, you can do so using:
        ```bash
        pip install black
        Once installed, run Black on your code:
        ```bash
        black .
        Our CI (Continuous Integration) pipeline checks that all code changes comply with the Black formatting. Make sure your code passes these checks before opening a pull request.
        For more details on Black and its configuration options, refer to the https://black.readthedocs.io/en/stable/
        We appreciate your efforts in maintaining a consistent and clean codebase!
    
License:
    MIT license

Acknowledgements:
    I would like to express my gratitude to the following individuals and resources that have contributed to the development and success of this project:

    Libraries and Tools
        - [Streamlit](https://docs.streamlit.io/): An Open-source Python library, which enables developers to build attractive user interfaces in no time.
        - [Mlflow](https://mlflow.org/docs/latest/index.html): An open source platform for the end-to-end machine learning lifecycle. A tracking API and UI
        - [Sphinx](https://www.sphinx-doc.org/en/master/index.html): An open source lib.  easy to create intelligent and beautiful documentation.
        - [Scikit-learn](https://scikit-learn.org/0.21/documentation.html): An open source machine learning lib.
        - [kaggle](https://www.kaggle.com/docs):Kaggle is the world's largest data science community with powerful tools and resources to help you achieve your data science goals

Inspiration
- Blogs from Medium, geeksforgeeks, Analytics vidya, Stack overflow and many more

Contact Information:
    https://github.com/ArunKhare 

Changelog:
    [Unreleased]
    - deployment in AWS  and Snowflake

    [Version 1.0.0] - 07-01-20024
    - [Version 1.0.0]: <Link to the release page or commit>

Roadmap:
    - Multicluster depolyment along with scheduling-Airflow and streaming pipeline-Kafka
    - converting Python code to Pyspark


Certainly! Here's a formatted version of your README for the **FraudDetection** project:

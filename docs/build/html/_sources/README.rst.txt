
Description
-----------

**FraudDetection** (Fraud Detection)is Package built in Python for fraud transactions in a banking system using supervised machine learning. It pulls data from the `Kaggle database <https://www.kaggle.com/>`_ and offers a *simple* and *intuitive* solution.


Purpose
-------
To detect the fraud transaction in the customers account 

Badges
------

.. image:: https://img.shields.io/badge/GitHub-ArunKhare/FraudDetection.git-blue.svg
    :target: https://github.com/ArunKhare/FraudDetection.git
    :alt: GitHub

.. image:: https://img.shields.io/badge/License-MIT-green.svg
    :target: https://opensource.org/licenses/MIT
    :alt: MIT License

Table of Contents
-----------------

- `Installation <#installation>`_
- `Usage <#usage>`_
- `Contributing <#contributing>`_
- `Coding Standards <#coding-standards>`_
- `License <#license>`_
- `Acknowledgements <#acknowledgements>`_
- `Changelog <#changelog>`_
- `Roadmap <#roadmap>`_


.. _installation:

Installation
------------

Create a Conda environment in root directory of the project:
    - This will install Python=3.10 and  all the dependencies from *'requirements_dev.txt'* for the project:

.. code-block:: shell

    bash init_setup.sh

Instructions to run the project:

.. code-block:: shell

    conda activate ./FraudDetection

.. _usage:

Usage
-----

For running the project for training and predictions from the root directory:

Usage scenarios:

1. From the console:
    - Training:
      ``python apps/app.py``
    - Prediction:
      ``python src/fraudDetection/components/prediction/prediction_service.py``

2. Streamlit app:
    ``python apps/trainingapp.py``

    - Run 'streamlit run <path>/trainingapp.py'
    - This will run the Streamlit server in the background.
    - You can now view your Streamlit app in your browser:
        - Local URL: http://localhost:8501
        - Network URL: http://192.168.99.138:8501

   A friendly UI will guide you through its various usage for training and prediction.

    screenshots

.. image:: _static/screenshots/StreamlitApp-cli.png
   :alt: Alt text for the screenshot
   :align: center

Alternatively, if you are editing the code:
    Run using:

.. code-block:: shell

    python dvc init
    python dvc repro

Running MLflow UI:

.. code-block:: shell

    mlflow ui

    - This runs the MLflow UI server in the background.
    - Click on link http://127.0.0.1:5000

    screenshot
.. image:: _static/screenshots/MlfowApp-cli.png
   :alt: Alt text for the screenshot
   :align: center

Environment Variable:

    - Use <root_dir>/.FraudDetection

.. code-block:: shell

    MLFLOW_TRACKING_URI=sqlite:///mlruns.db

For testing code:

FraudDetection Project is configured with pytest.

Configure your project for specific needs using configuration files:

    - tox.ini
    - pyproject.toml
    - setup.py

.. _contributing:

Contributing
------------

1. Links and Details:

   - `Good First Issue <https://github.com/ArunKhare/FraudDetection/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_
   - Fork, install the project as mentioned in the Installation section, test the code using pytest, and create a pull request.

2. Coding Standard

   Follow the `Black <https://github.com/psf/black>`_ code style for this project. Black is an opinionated code formatter that ensures consistent formatting across the codebase.

   To ensure code consistency and readability, we recommend running Black before submitting any code changes. If you haven't installed Black yet, you can do so using:

   .. code-block:: shell

       pip install black

   Once installed, run Black on your code:

   .. code-block:: shell

       black .

   Our CI (Continuous Integration) pipeline checks that all code changes comply with the Black formatting. Make sure your code passes these checks before opening a pull request.

   For more details on Black and its configuration options, refer to the `Black Documentation <https://black.readthedocs.io/en/stable/>`_.

   We appreciate your efforts in maintaining a consistent and clean codebase!

.. _license:

License
-------

MIT license

.. _acknowledgements:

Acknowledgements
----------------

I would like to express my gratitude to the following individuals and resources that have contributed to the development and success of this project:

Libraries and Tools:

- `Streamlit <https://docs.streamlit.io/>`_: An open-source Python library that enables developers to build attractive user interfaces in no time.
- `Mlflow <https://mlflow.org/docs/latest/index.html>`_: An open-source platform for the end-to-end machine learning lifecycle. A tracking API and UI.
- `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_: An open-source library easy to create intelligent and beautiful documentation.
- `Scikit-learn <https://scikit-learn.org/0.21/documentation.html>`_: An open-source machine learning library.
- `Kaggle <https://www.kaggle.com/docs>`_: Kaggle is the world's largest data science community with powerful tools and resources to help you achieve your data science goals.

Inspiration:

- Blogs from Medium, GeeksforGeeks

Contact Information
-------------------

    https://github.com/ArunKhare 

Changelog
---------

    [Unreleased]
    - deployment in AWS  and Snowflake

    [Version 1.0.0] - 07-01-20024
    - [Version 1.0.0]: <Link to the release page or commit>

Roadmap
-------
    - Multicluster depolyment along with scheduling-Airflow and streaming pipeline-Kafka
    - converting Python code to Pyspark
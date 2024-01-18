"""
This module defines functions related to launching *MLflow UI*.
"""
import os
import sys
import subprocess
import streamlit as st
from fraudDetection.exception import FraudDetectionException


def exp_tracking():
    """
    Displays a Streamlit app with a button to launch the MLflow UI.
    :return: None
    """

    st.sidebar.markdown(
        "This is a Streamlit app that integrates with MLflow to manage experiments and models."
    )

    st.sidebar.info(
        "Note: Make sure your MLflow Tracking Server is running or replace the URI in the code.",
        icon="ℹ️",
    )

    # Add a radio button to select the tracking URI type
    uri_type = st.radio("Select Tracking URI Type", ["MySQL", "SQLite"])

    if uri_type == "MySQL":
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI_MYSQL")
    else:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI_SQLITE", "sqlite:///mlruns.db")

    st.button(
        "MLflow UI",
        key="MLFLOW UI",
        on_click=lambda: launch_mlflow_ui(tracking_uri),
        type="primary",
    )


def launch_mlflow_ui(tracking_uri):
    """Function to launch the MLflow UI using subprocess.
    This function attempts to launch the MLflow UI using subprocess. It specifies the
    port and backend-store-uri parameters for configuring the UI.
    Args:
        tracking_uri (str): The MLflow tracking URI to use.
    Returns:
        None
    """
    try:
        # Launch MLflow UI using subprocess with the specified tracking URI
        subprocess.Popen(
            [
                "mlflow",
                "ui",
                "--port",
                "8080",
                "--backend-store-uri",
                tracking_uri,
            ]
        )
    except subprocess.CalledProcessError as e:
        st.error(f"Error launching MLflow UI: {e}")
        raise FraudDetectionException(e, sys) from e

    with st.container(border=True):
        st.info(
            "Click the HERE :red[*'http://127.0.0.1:8080'*] to open Mlfow UI in browser Or incase of windows :red[*'http://localhost:8080'*] ",
            icon="ℹ️",
        )
        st.link_button("Go to Mlflow UI", "http://127.0.0.1:8080")

        st.toast("MLflow UI is now running in the background", icon="🎉")

"""
This module defines functions related to launching *MLflow UI*.
"""
import sys
import subprocess
import streamlit as st
from fraudDetection.exception import FraudDetectionException


def exp_tracking():
    """
    Displays a Streamlit app with a
    button to launch the MLflow UI.
    :return: None
    """

    st.sidebar.markdown(
        "This is a Streamlit app that integrates with MLflow to manage experiments and models."
    )

    st.sidebar.info(
        "Note: Make sure your MLflow Tracking Server is running or replace the URI in the code.",
        icon="ℹ️",
    )

    st.button("MLflow UI", key="MLFLOW UI", on_click=launch_mlflow_ui, type="primary")


def launch_mlflow_ui():
    """Function to launch the MLflow UI using subprocess.
    This function attempts to launch the MLflow UI using subprocess. It specifies the
    port and backend-store-uri parameters for configuring the UI.
    Args:
        None
    Returns:
        None
    """
    try:
        # Launch MLflow UI using subprocess
        subprocess.Popen(
            [
                "mlflow",
                "ui",
                "--port",
                "8080",
                "--backend-store-uri",
                "sqlite:///mlruns.db",
            ]
        )
    except subprocess.CalledProcessError as e:
        st.error(f"Error launching MLflow UI: {e}")
        raise FraudDetectionException(e, sys) from e

    with st.container(border=True):
        st.info(
            "Click the HERE :red[*'http://127.0.0.1:8080'*] to open Mlfow UI in browser",
            icon="ℹ️",
        )
        st.link_button("Go to Mlflow UI", "http://127.0.0.1:8080")

        st.toast("MLflow UI is now running in back ground", icon="🎉")

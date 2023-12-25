
import streamlit as st
import mlflow
import pandas as pd
import os
import logging
from dotenv import load_dotenv
import subprocess
from pathlib import Path
import threading
import webbrowser


def exp_tracking():
    
    load_dotenv()
    
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default="http://localhost:5000")
    
    if not mlflow_tracking_uri:
        st.error("MLFLOW_TRACKING_URI not set.")
        logging.error("MLFLOW_TRACKING_URI not set.")
        return

    st.sidebar.markdown("This is a Streamlit app that integrates with MLflow to manage experiments and models." )
  
    st.sidebar.info("Note: Make sure your MLflow Tracking Server is running or replace the URI in the code.",icon="ℹ️")
    
    st.button("MLflow UI", key="MLFLOW UI", on_click=launch_mlflow_ui, type='primary')
    
def launch_mlflow_ui():
        
        try:
            # Launch MLflow UI using subprocess
            subprocess.Popen(["mlflow", "ui", "--port", "8080", "--backend-store-uri", "sqlite:///mlruns.db"])
        except subprocess.CalledProcessError as e:
            st.error(f"Error launching MLflow UI: {e}")
        st.info("Click the server uri :red['http://127.0.0.1:8080'] in console to open Mlfow UI in browser", icon="ℹ️")
        st.toast("MLflow UI is now running in back ground", icon='🎉')

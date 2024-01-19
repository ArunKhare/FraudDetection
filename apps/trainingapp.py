"""
This module defines the `FraudDetectionTrainingApp` class, which facilitates the training of machine learning models for fraud detection. It utilizes a pipeline structure for model training, MLflow for experiment tracking, and Streamlit for creating a user-friendly interface.

The application allows users to:
- Update existing model configurations.
- Create entirely new machine learning model configurations.
- Validate configurations using a JSON formatter within the UI.
- View and compare experiments related to model training.
- Access data and logs for in-depth analysis.
- Utilize a single-click feature to navigate to the MLflow tracking UI, enabling users to view, compare, and download model artifacts such as 'model.pickle', 'Signature', 'Processing object', and 'Requirements'.
- Employ a prediction service that supports model predictions, with the capability to upload a batch file or input a single record for processing.

Environment Variables:
- MLFLOW_TRACKING_URI_SQLITE: The URI used by MLflow for experiment tracking, loaded from the environment.
- MLFLOW_TRACKING_URI_MYSQL: The URI used by MLflow for experiment tracking in cloud server, loaded from the environment.

"""

import os
import sys
import json
from pathlib import Path
import yaml
import pandas as pd
import streamlit as st
import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import InvalidUrlException
from fraudDetection.pipeline.pipeline import Pipeline
from fraudDetection.config.configuration import (
    ConfigurationManager,
    ROOT_DIR,
    CONFIG_FILE_PATH,
)
from fraudDetection.utils import read_yaml, write_yaml
from fraudDetection.constants import CONFIG_DIR
from fraudDetection.components import FraudDetectionPredictorApp
from fraudDetection.logger import get_log_dataframe, logging
from fraudDetection.exception import FraudDetectionException
from mlflowapp import exp_tracking


load_dotenv()


def set_tracking_uri():
    """Set Mlflow Tracking URI
    Raises:
        InvalidUrlException: when URI is not in proper format
    """
    mlflow_tracking_uri = os.getenv(
        "MLFLOWTRACKINGURI", default="http://localhost:8080"
    )
    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    except InvalidUrlException as e:
        st.error("MLFLOW_TRACKING_URI not set.")
        raise FraudDetectionException(e, sys) from e
    finally:
        logging.info(f"Tracking URI set {mlflow_tracking_uri}")


# # LOCAL RUN
# conn_location = Path(DATA_INGESTION_KAGGLE_CONFIG_FILE_PATH)
# connect = load_json(conn_location)
# os.environ["KAGGLE_CONFIG_DIR"] = str(conn_location.parent)
# os.environ["KAGGLE_USERNAME"] = connect["username"]
# os.environ["KAGGLE_KEY"] = connect["key"]
# print("No specific deployment check provided. Performing default actions...")


class Constants:
    """This class holds constant values for the module."""

    LOG_FOLDER_NAME = "logs"
    PIPELINE_FOLDER_NAME = "FraudDetection"
    SAVED_MODELS_DIR_NAME = "saved_models"
    MODEL_CONFIG_FILE_PATH = ROOT_DIR / CONFIG_DIR / "model.yaml"
    LOG_DIR = ROOT_DIR / LOG_FOLDER_NAME
    PIPELINE_DIR = ROOT_DIR / PIPELINE_FOLDER_NAME
    MODEL_DIR = ROOT_DIR / SAVED_MODELS_DIR_NAME
    ARCHIVE_FOLDER = ROOT_DIR / CONFIG_DIR / "archive"


class FraudDetectionTrainingApp:
    """This class defines the main training application for fraud detection."""

    def __init__(self) -> None:
        """Initialize FraudDetectionTrainingApp.
        Initializes attributes for configuration and pipeline."""

        self.config = ConfigurationManager(config=CONFIG_FILE_PATH)
        self.pipeline = None

    def initialize_pipeline(self):
        """Initialize the pipeline for training."""
        self.pipeline = Pipeline(config=self.config)

    def get_pipeline_artifact(self):
        """Get the artifact directory for the training pipeline.
        Returns:
            str: Path to the artifact directory."""
        try:
            training_pipeline_artifact = self.config.get_training_pipeline_config
            artifact_dir = training_pipeline_artifact.artifacts_root
            os.makedirs(artifact_dir, exist_ok=True)

        except Exception as e:
            st.exception(e)
            logging.info(e)
        return artifact_dir

    def main(self):
        """Main method to run the training application."""
        st.set_page_config(
            page_title="Fraud Detection System Training",
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={"About": "# This is a Fraud Detection System! by Arun khare"},
        )

        if "counter" not in st.session_state:
            st.session_state.counter = 0

        st.sidebar.title("Navigation")
        selected_page = st.sidebar.radio(
            "Go to",
            [
                "Home",
                "Artifacts",
                "Logs",
                "Experiments",
                "Training",
                "Model Configs",
                "Mlflow Tracking",
                "Prediction Service",
            ],
        )

        artifact_path = self.get_pipeline_artifact()

        try:
            if selected_page == "Home":
                st.title("🎯 :rainbow[Training Index Page]")
                st.divider()
                self.display_home()

            elif selected_page == "Artifacts":
                st.title("📀:orange[Artifacts]")

                hierarchy_dict = FraudDetectionTrainingApp.generate_hierarchy_dict(
                    artifact_path
                )
                self.display_hierarchy(hierarchy_dict, artifact_path)

            elif selected_page == "Logs":
                st.title("✉️ :orange[Log Message]")
                st.divider()
                self.display_logs()

            elif selected_page == "Experiments":
                st.title("🧮 :orange[Experiments]")
                st.divider()
                self.view_experiment_history()

            elif selected_page == "Model Configs":
                st.title("✍️ :orange[Model Configuration Editor]")
                st.divider()
                self.update_model_config()

            elif selected_page == "Training":
                st.title(" :orange[Train Models]")
                st.divider()
                self.train()
                st.session_state.counter += 1

            elif selected_page == "Mlflow Tracking":
                st.title("✍️ :orange[MLflow Integration with Streamlit]")
                st.divider()
                exp_tracking()

            elif selected_page == "Prediction Service":
                FraudDetectionTrainingApp.predict()

        except Exception as e:
            st.exception("An error occurred. Please check the logs for more details")
            logging.info(e)
            raise FraudDetectionException(e, sys) from e

    def display_home(self):
        """Display the home page."""
        st.write("Welcome to the Home Page!")

    @staticmethod
    def generate_hierarchy_dict(root_path):
        """Generate a hierarchy dictionary for file structure.
        Args:
            root_path (str): Root path for file structure.
        Returns:
            dict: Hierarchy dictionary."""
        try:
            hierarchy_dict = {}

            for item in os.listdir(root_path):
                item_path = root_path / item

                if os.path.isdir(item_path):
                    # Recursively call the function for subdirectories
                    sub_dict = FraudDetectionTrainingApp.generate_hierarchy_dict(
                        item_path
                    )

                    # Merge common prefixes in keys
                    if sub_dict is not None:
                        for key, value in sub_dict.items():
                            full_key = Path(os.path.join(item, key))
                            hierarchy_dict[full_key] = value
                elif os.path.isfile(item_path):
                    # If it's a file, add it to the dictionary
                    hierarchy_dict[item] = [item]
            return hierarchy_dict
        except ValueError as e:
            st.error(
                """An error occurred while generating the file structure in to a dict.
                Please check the logs for more details"""
            )
            logging.info(e)
            return {}

    def display_hierarchy(self, hierarchy_dict, current_path):
        """Display the hierarchy of artifacts and content of files in tabs.
        Args:
            hierarchy_dict (dict): Hierarchy dictionary.
            current_path (str): Current path for artifacts.
        """
        try:
            current_path_sub_dir = os.listdir(current_path)
            tabs = st.tabs(current_path_sub_dir)
            parent_folder_name = os.path.basename(current_path)

            dict_paths = {}
            for item in hierarchy_dict:
                key = os.path.dirname(item)
                if key not in dict_paths:
                    dict_paths[key] = [os.path.basename(item)]
                else:
                    dict_paths[key].append(os.path.basename(item))

            # Display parent folder name as header
            with st.container(border=True):
                # Display files and subdirectories under the parent folder
                st.title(f"📁 _Parent Folder_: {parent_folder_name}")

                for i, tab in enumerate(tabs):
                    with tab:
                        st.header(f"{current_path_sub_dir[i]}".capitalize())
                        temp = []
                        if len(temp) > 0:
                            for key in temp:
                                dict_paths.pop(key, None)
                        for key, value in dict_paths.items():
                            sub_dirs = [*key.split(os.path.sep)]
                            if current_path_sub_dir[i] == sub_dirs[0]:
                                temp.append(key)
                                if isinstance(value, list):
                                    for file in value:
                                        file_path = Path(
                                            os.path.join(current_path, key, file)
                                        )
                                        if st.button(f"📄 File: {file}", key=file_path):
                                            FraudDetectionTrainingApp.display_file_content(
                                                file_path
                                            )

                                    if len(sub_dirs) > 1:
                                        st.subheader(
                                            ":blue[Subdirectories] :herb:",
                                            divider="rainbow",
                                        )
                                        for subdir in sub_dirs[1:]:
                                            st.write(f"{subdir}/")
        except Exception as e:
            st.error(
                "Error while displaying artifact heirarchy pl check logs for details"
            )
            logging.info(e)
            raise FraudDetectionException(e, sys) from e

    @staticmethod
    def display_file_content(file_path):
        """Display the file content .
        Args:
            file_path (str)"""

        file_path = Path(file_path)

        try:
            file_extension = file_path.suffix.lower()

            if file_extension == ".csv":
                # Display content of CSV file
                # with open(file_path, "r") as file:
                #     file_contents = file.read()
                df = pd.read_csv(file_path)
                # st.text_area(f"Contents of {file_path}", file_contents, height=300)
                st.dataframe(df)
            elif file_extension in [".json", ".yaml"]:
                # Display content of JSON or YAML file
                with open(file_path, "r") as file:
                    file_contents = file.read()
                    st.text_area(f"Contents of {file_path}", file_contents, height=300)
            elif file_extension in [".pkl", ".npz"]:
                # Display full file path for models or npz files
                st.info(f"Full File Path: {file_path}")
            else:
                st.warning("Unsupported file type. Click to see full file path.")
                st.info(f"Full File Path: {file_path}")
        except Exception as e:
            st.error(f"Error while displaying file content: {e}")
            raise FraudDetectionException(e, sys) from e

    def view_experiment_history(self):
        """View the experiment history."""
        try:
            experiment_df = Pipeline.get_experiment_status()

            if experiment_df.empty:
                st.warning("No experiment data available.")
            else:
                st.write(experiment_df)
                st.header("Experiment Stats")
                st.write(experiment_df.describe())

        except Exception as e:
            st.exception(
                f"An error occurred. Please check the logs for details. Error: {e}"
            )
            logging.info(e)
            raise FraudDetectionException(e, sys) from e

    def train(self):
        """Train models using the pipeline"""
        message = ""

        if st.button("⚔️start training", type="primary"):
            with st.spinner("Training in progess............"):
                st.session_state
                pipeline = Pipeline(config=ConfigurationManager(CONFIG_FILE_PATH))
                if not Pipeline.experiment.running_status:
                    message = "Training started."
                    st.session_state.message = message
                    pipeline.run()
                else:
                    message = "Training is already in progress."
                    st.session_state.message = message

    @staticmethod
    def predict():
        """Run the FraudDetectionPredictorApp for predictions."""
        FraudDetectionPredictorApp.run()

    def update_model_config(self):
        """Update the model configuration."""
        try:
            # Load existing configuration
            current_config = read_yaml(Constants.MODEL_CONFIG_FILE_PATH)
            toggle_function = st.toggle(
                ":red[__Edit configuration__ 🌷]", key="edit configuration"
            )

            if toggle_function:
                FraudDetectionTrainingApp.edit(current_config)
            else:
                FraudDetectionTrainingApp.display_configuration(current_config)

        except Exception as e:
            st.error(
                """An error occured may be the file format is not correct. 
                pl check the logs for details"""
            )
            logging.info(e)
            return str(e)

    @staticmethod
    def display_configuration(config):
        """Display the current model configuration.
        Args:
            config: Model configuration.
        """
        st.write("Current Configuration:")
        with st.container(border=True):
            st.code(json.dumps(config, indent=2), language="json")

    @staticmethod
    def edit(config):
        """Edit the model configuration.
        Args:
            config: Model configuration."""

        # User can edit the configuration
        st.write("Edit Configuration:")
        with st.container(border=True):
            edited_config = st.text_area(
                "Edit the YAML configuration", json.dumps(config, indent=2), height=1000
            )

        # Validate the edited configuration using the JSON formatter
        try:
            validated_config = yaml.safe_load(edited_config)
            st.success("Configuration is valid!")
        except Exception as e:
            st.exception(e)
            logging.info(e)
            return str(e)

        # Save the updated configuration
        write_yaml(Constants.MODEL_CONFIG_FILE_PATH, validated_config)

        # Archive the current configuration with a version name
        version_name = st.text_input(":red[__Enter version name for archiving:__]")

        if version_name:
            # archive_config(current_config, version_name)
            archive_path = (
                Constants.ARCHIVE_FOLDER / f"model_config_{version_name}.yaml"
            )
            write_yaml(archive_path, config)
            st.success(f"Configuration archived as version: {version_name}")
        else:
            st.error("Yaml format error")

    def display_logs(self):
        """Display logs in the logs directory."""
        try:
            logs_path = ROOT_DIR / Constants.LOG_FOLDER_NAME

            os.makedirs(logs_path, exist_ok=True)
            FraudDetectionTrainingApp.delete_empty_files(Constants.LOG_DIR)
            logging.info(f"logs_Path: {logs_path}")

            if not os.path.exists(logs_path):
                return st.warning("'logs' directory Path does not exist")
            # Show directory contents
            files_path = [
                os.path.join(logs_path, file) for file in os.listdir(logs_path)
            ]
            # display log file
            log_file_selected = st.selectbox(
                "Log files", files_path, label_visibility="hidden"
            )

            log_df = get_log_dataframe(log_file_selected)
            with st.container(border=True):
                st.dataframe(log_df, use_container_width=True, height=500)

        except Exception as e:
            st.exception("An error occured pl check the logs for details")
            logging.info(e)
            raise FraudDetectionException(e, sys) from e

    @staticmethod
    def delete_empty_files(directory):
        """Delete empty files in a directory.
        Args:
            directory (str): Directory path.
        Returns:
            None
        """
        for filename in os.listdir(directory):
            file_path = Path(os.path.join(directory, filename))

            # FraudDetectionTraningApp.remove_blank_lines(file_path)

            # Check if the file is empty
            if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
                logging.info("Deleting empty log files")
                os.remove(file_path)


def init_app():
    """Initialize training and prediction app"""
    set_tracking_uri()
    instance_training = FraudDetectionTrainingApp()
    instance_training.initialize_pipeline()
    instance_training.main()


if __name__ == "__main__":
    # streamlit_community_cloud()
    init_app()

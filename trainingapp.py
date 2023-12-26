from matplotlib.style import context

import os, sys
import json
import yaml
from pathlib import Path
from fraudDetection.pipeline.pipeline import Pipeline
from fraudDetection.config.configuration import ConfigurationManager, ROOT_DIR, CONFIG_FILE_PATH
from fraudDetection.utils import read_yaml, write_yaml
from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging
from fraudDetection.constants import CONFIG_DIR, get_current_time_stamp
from fraudDetection.pipeline.pipeline import Pipeline
from fraudDetection.components import FraudDetectionPredictorApp
from fraudDetection.logger import get_log_dataframe
from fraudDetection.utils import read_yaml, write_yaml
from mlflowapp import exp_tracking
import streamlit as st

ROOT_DIR = Path(os.getcwd())
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "FraudDetection"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = Path(os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml"))
LOG_DIR = Path(os.path.join(ROOT_DIR, LOG_FOLDER_NAME))
PIPELINE_DIR = Path(os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME))
MODEL_DIR = Path(os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME))
ARCHIVE_FOLDER = Path(os.path.join(ROOT_DIR, CONFIG_DIR,"archive"))


class FraudDetectionTraningApp():
    
    def __init__(self) -> None:
        self.config = ConfigurationManager(config=CONFIG_FILE_PATH,current_time_stamp=get_current_time_stamp())
        self.pipeline = Pipeline(config=self.config)

    def get_pipline_artifact(self):
        try:

            training_pipeline_artifact = self.config.get_training_pipeline_config
            artifact_dir = training_pipeline_artifact.artifacts_root
            os.makedirs(artifact_dir, exist_ok=True)

        except Exception as e:
            st.exception(e)
            logging.exception(e)
        return artifact_dir
    
    
    def main(self):

        st.set_page_config(
            page_title="Fraud Detection System Training",
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={'About': "# This is a Fraud Detection System! by Arun khare"}
            )
        
        if 'counter' not in st.session_state:
            st.session_state.counter = 0
        
        st.sidebar.title("Navigation")
        selected_page = st.sidebar.radio("Go to", ["Home", "Artifacts", "Logs", "Experiments", "Training", "Model Configs", "Mlflow Tracking", "Prediction Service"])
        
        artifact_path = self.get_pipline_artifact()

        try:

            if selected_page == "Home":
                st.title('🎯 :rainbow[Training Index Page]')
                st.divider()
                self.display_home()

            elif selected_page == "Artifacts":
                st.title('📀:orange[Artifacts]')
                
                hierarchy_dict = FraudDetectionTraningApp.generate_hierarchy_dict(artifact_path)
                self.display_hierarchy(hierarchy_dict, artifact_path)

            elif selected_page == "Logs":
                st.title('✉️ :orange[Log Message]')
                st.divider()
                self.display_logs()

            elif selected_page == "Experiments":
                st.title('🧮 :orange[Experiments]')
                st.divider()
                self.view_experiment_history()

            elif selected_page == "Model Configs":
                st.title('✍️ :orange[Model Configuration Editor]')
                st.divider()
                self.update_model_config()
            
            elif selected_page == "Training":
                st.title(' :orange[Train Models]')
                st.divider()
                self.train()
                st.session_state.counter += 1

            elif selected_page == "Mlflow Tracking":
                st.title("✍️ :orange[MLflow Integration with Streamlit]")
                st.divider()
                exp_tracking()

            elif selected_page == "Prediction Service":
                FraudDetectionTraningApp.predict()

        except Exception as e:
            st.exception(e)
            logging.exception(e)
           
    def display_home(self):
        st.write("Welcome to the Home Page!")
    
   
    @staticmethod       
    def generate_hierarchy_dict(root_path):

        try:
            hierarchy_dict = {}

            for item in os.listdir(root_path):
                item_path = os.path.join(root_path, item)

                if os.path.isdir(item_path):
                    # Recursively call the function for subdirectories
                    sub_dict = FraudDetectionTraningApp.generate_hierarchy_dict(item_path)

                    # Merge common prefixes in keys
                    if sub_dict is not None:
                        for key, value in sub_dict.items():
                            full_key = os.path.join(item, key)
                            hierarchy_dict[full_key] = value
                elif os.path.isfile(item_path):
                    # If it's a file, add it to the dictionary
                    hierarchy_dict[item] = [item]
        except Exception as e:
            st.error('An error occured while generating the file structure in to a dict{e}')
            logging.exception(e)
        return hierarchy_dict

    def display_hierarchy(self, hierarchy_dict, current_path):
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
                st.title(f'📁 _Parent Folder_: {parent_folder_name}')

                for i, tab in enumerate(tabs):
                    with tab:
                        st.header(f'{current_path_sub_dir[i]}'.capitalize())
                        temp = []
                        if len(temp):
                            for key in temp:
                                dict_paths.pop(key, None)
                        for key, value in dict_paths.items():
                            sub_dirs = [*key.split(os.path.sep)]
                            if current_path_sub_dir[i] == sub_dirs[0]:
                                temp.append(key)
                                if isinstance(value, list):
                                    for file in value:
                                        file_path = os.path.join(current_path, key, file)
                                        if st.button(f"📄 File: {file}", key=file_path):
                                            FraudDetectionTraningApp.display_file_content(file_path)

                                    if len(sub_dirs) > 1:
                                        st.subheader(":blue[Subdirectories] :herb:", divider= 'rainbow')
                                        for subdir in sub_dirs[1:]:
                                            st.write(f"{subdir}/")
        except Exception as e:
            st.error(f"Error while displaying artifact heirarchy: {e}")                        
            logging.exception(e)
                            
    @staticmethod
    def display_file_content(file_path):
        file_path = Path(file_path)

        try:
            file_extension = file_path.suffix.lower()

            if file_extension == '.csv':
                # Display content of CSV file
                with open(file_path, 'r') as file:
                    file_contents = file.read()
                    st.text_area(f"Contents of {file_path}", file_contents, height=300)
            elif file_extension in ['.json', '.yaml']:
                # Display content of JSON or YAML file
                with open(file_path, 'r') as file:
                    file_contents = file.read()
                    st.text_area(f"Contents of {file_path}", file_contents, height=300)
            elif file_extension in ['.pkl', '.npz']:
                # Display full file path for models or npz files
                st.info(f"Full File Path: {file_path}")
            else:
                st.warning("Unsupported file type. Click to see full file path.")
                st.info(f"Full File Path: {file_path}")
        except Exception as e:
            st.error(f"Error while displaying file content: {e}")

    def view_experiment_history(self):
        try:
        
            experiment_df = Pipeline.get_experiment_status()
            st.write(experiment_df)
            st.header("Experiment Stats")
            st.write(experiment_df.describe())

        except Exception as e:
            st.exception(e)
            logging.exception(e)

    def train(self):
        message = ""
        
        if st.button('⚔️start training', type="primary"):
            with st.spinner("Training in progess............"):
                st.session_state
                pipeline = Pipeline(config= ConfigurationManager(current_time_stamp=get_current_time_stamp()))
                if not Pipeline.experiment.running_status:
                    message = "Training started."
                    st.session_state.message = message
                    pipeline.run()
                else:
                    message = "Training is already in progress."
                    st.session_state.message = message
    
    @staticmethod
    def predict():
        FraudDetectionPredictorApp.run()

    def update_model_config(self):

        try:
            # Load existing configuration
            current_config = read_yaml(MODEL_CONFIG_FILE_PATH)
            toggle_function = st.toggle(":red[__Edit configuration__ 🌷]",key = "edit configuration")

            if toggle_function:

                FraudDetectionTraningApp.edit(current_config)
            else:
                   
                FraudDetectionTraningApp.display_configuration(current_config)

        except  Exception as e:
            logging.exception(e)
            return str(e)
        
    @staticmethod
    def display_configuration(config):

        # Display current configuration
        st.write("Current Configuration:")
        with st.container(border=True):
            st.code(json.dumps(config, indent=2), language="json")

    @staticmethod
    def edit(config):
                
                # User can edit the configuration
                st.write("Edit Configuration:")
                with st.container(border=True):
                    edited_config = st.text_area("Edit the YAML configuration", json.dumps(config, indent=2), height=1000)
                
                # Validate the edited configuration using the JSON formatter
                try:
                    validated_config = yaml.safe_load(edited_config)
                    st.success("Configuration is valid!")
                except Exception as e:
                    st.exception(e)
                    logging.exception(e)
                    return str(e)

                # Save the updated configuration
                write_yaml(MODEL_CONFIG_FILE_PATH,validated_config)

                # Archive the current configuration with a version name
                version_name = st.text_input(":red[__Enter version name for archiving:__]")

                if version_name:
                    # archive_config(current_config, version_name)
                    archive_path = ARCHIVE_FOLDER / f"model_config_{version_name}.yaml"
                    write_yaml(archive_path, config)
                    st.success(f"Configuration archived as version: {version_name}")
                else:
                    st.e

    def display_logs(self):
        try:
                
            logs_path = Path(os.path.join(ROOT_DIR,LOG_FOLDER_NAME))
       
            os.makedirs(logs_path, exist_ok=True)
            FraudDetectionTraningApp.delete_empty_files(LOG_DIR)
            logging.info(f"logs_Path: {logs_path}")
            
            if not os.path.exists(logs_path):
                return st.warning("'logs' directory Path does not exist") 
            # Show directory contents
            files_path = [os.path.join(logs_path, file) for file in os.listdir(logs_path)]
            #display log file
            log_file_selected = st.selectbox(f'Log files', files_path,label_visibility='hidden')
    
            log_df = get_log_dataframe(log_file_selected)
            with st.container(border=True):
                st.dataframe(log_df,use_container_width=True,height=500)
                     
        except Exception as e:
            st.exception(e)
            logging.exception(e)

    @staticmethod
    def delete_empty_files(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # FraudDetectionTraningApp.remove_blank_lines(file_path)

            # Check if the file is empty
            if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
                logging.info("Deleting empty log files")
                os.remove(file_path)

if __name__ == "__main__":
   instancetraining =  FraudDetectionTraningApp()
   instancetraining.main()

    

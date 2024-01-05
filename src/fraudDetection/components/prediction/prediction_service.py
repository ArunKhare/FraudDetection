"""
This module provides classes and functions for fraud detection in banking transactions. It includes
a FraudDetectionModel class for loading and using the trained model, and a FraudDetectionPredictorApp
class for creating a Streamlit web application for making predictions.

Module Constants:
    - tags (dict): Dictionary containing version information.
    - INPUT_SCHEMA_JSON_FILE_NAME (str): Name of the input schema JSON file.
    - MODEL_EVALUATION_YAML (str): Name of the model evaluation YAML file.
    - SAVED_MODELS_DIR (Path): Directory path for saving models.
    - MODEL_NAME (str): Name of the model file.
    - PREPROCESSER_OBJ_PATH (Path): Path to the preprocessed data object.
    - EVALUATED_MODELS_PATH (Path): Path to the evaluated models.

Classes:
    - FraudDetectionModel: Class for loading and using the trained model.
    - FraudDetectionPredictorApp: Class for creating a Streamlit web application for predictions.

Usage:

if __name__ == "__main__":
    FraudDetectionPredictorApp.run()
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import joblib
import json
import re
import streamlit as st

tags = {"Version": "v1"}
INPUT_SCHEMA_JSON_FILE_NAME = "input_schema.json"
MODEL_EVALUATION_YAML = "model_evaluation.yaml"
SAVED_MODELS_DIR = Path("artifacts\saved_models")
MODEL_NAME = "model.pkl"
PREPROCESSER_OBJ_PATH = Path(
    r"artifacts\transformed_data\preprocessed\preprocessed.pkl"
)
EVALUATED_MODELS_PATH = Path(r"artifacts\model_evaluation")


# schema_file_path = Path(r"C:\Users\arunk\FraudDetection\configs\schema.yaml")
class FraudDetectionModel:
    """This class provides methods for loading the trained model, getting input schema,
    transforming input data, and making predictions.
     Methods:
        - get_latest_model(): Load the latest trained model from evaluated models.
        - get_frauddetection_input_schema(): Get the input schema for fraud detection.
        - transform_input(df): Transform input DataFrame using preprocessed data object.
        - predict(df): Make predictions using the loaded model.
    """

    def __init__(self) -> None:
        """Initializes attributes for model name, evaluated models path, and preprocessed data path."""
        self.model_name = MODEL_NAME
        self.evaluated_models_path = EVALUATED_MODELS_PATH
        self.preprocessd_obj_path = PREPROCESSER_OBJ_PATH

    def get_latest_model(self):
        """Get the latest trained model from evaluated models.
        Returns:
            Trained model object.
        Raises:
            ValueError: If there is an issue during the process.
        """
        try:
            """Load best model from the saved models and the input schema"""
            self.evaluated_models_file_path = Path(
                os.path.join(self.evaluated_models_path, MODEL_EVALUATION_YAML)
            )

            with open(self.evaluated_models_file_path, "r") as f:
                model_registry_paths = yaml.load(f, Loader=yaml.Loader)

            # model_registry_paths = read_yaml(self.evaluated_models_file_path)
            path_object = model_registry_paths["best_model"]["model_path"]

            # Convert the WindowsPath object to a string
            path_string = str(path_object)
            # Define a regex pattern for matching the date
            date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}")
            # Search for the date pattern in the string
            match = date_pattern.search(path_string)
            # Extract the matched date
            if match:
                best_model_name_dir = match.group()

            matching_dirs = SAVED_MODELS_DIR.glob(best_model_name_dir)
            # Check if any matching directories were found
            if matching_dirs:
                # matching directory
                saved_model_path = next(matching_dirs, None)
                model_file = Path(os.path.join(saved_model_path, self.model_name))
                model = joblib.load(model_file)
        except (ValueError, TypeError) as e:
            print(f"Error: {e}")
        except Exception as e:
            print(e)

        return model

    def get_frauddetection_input_schema(self):
        """Get the input schema for fraud detection.
        Returns:
            schema (dict): Input schema
        Raises:
            ValueError: If there is an issue during the process.
        """
        self.input_schema_path = Path(
            os.path.join(self.evaluated_models_path, INPUT_SCHEMA_JSON_FILE_NAME)
        )
        with open(self.input_schema_path, "r") as json_file:
            schema = json.load(json_file)
        return schema

    def transform_input(self, df) -> np.array:
        """Transform input DataFrame using preprocessed data object.
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            transformed_test_arr (Numpy.array): Transformed input
        Raises:
            ValueError: If there is an issue during the process.
        """
        try:
            """Load the pipeline object"""
            preprocessing_pipe_obj = joblib.load(self.preprocessd_obj_path)
            schema = self.get_frauddetection_input_schema()
            df_input = df[schema["columns"]]
            transformed_test_arr = preprocessing_pipe_obj.transform(df_input)
            return transformed_test_arr
        except (ValueError, TypeError) as e:
            print(f"Error: {e}")
        except Exception as e:
            print(e)

    def predict(self, df) -> pd.DataFrame:
        """Make predictions using the loaded model.
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            df (pd.DataFrame): DataFrame with predictions.
        Raises:
            ValueError: If there is an issue during the process.
        """
        try:
            model = self.get_latest_model()
            transformed_input = self.transform_input(df)
            prediction = model.predict(transformed_input)
            drop_features = ["isFlaggedFraud", "isFraud"]
            for col in drop_features:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            prediction_column_name = "Prediction isFraud"
            df[prediction_column_name] = prediction

            return df
        except (ValueError, TypeError) as e:
            print(f"Error: {e}")
        except Exception as e:
            print(e)


class FraudDetectionPredictorApp:
    """This class provides a Streamlit web application for making predictions using the FraudDetectionModel.
    Methods:
        - run(): Run the Streamlit application.
    Usage:
        if __name__ == "__main__":
            FraudDetectionPredictorApp.run()
    """

    @staticmethod
    def run():
        """Creates a Streamlit web application with sections for entering a single record or uploading a file
        to predict fraudulent transactions."""
        # st.set_page_config(
        #     page_title="Fraud Detection System",
        #     page_icon="🧊",
        #     layout="wide",
        #     initial_sidebar_state="expanded",
        #     menu_items={'About': "# This is a Fraud Detection System! by Arun khare"}
        #     )

        model = FraudDetectionModel()

        page = st.sidebar.radio(
            """ Hello there! I’ll guide you! Prediction Service""",
            [
                "Main Page",
                "Prediction",
            ],
            key="current_page",
        )

        if page == "Main Page":
            st.title("🏦 :rainbow[Hello, welcome to Fraud Detector]")
            st.caption(
                "This detects :blue[_Fraud Transaction_] :sunglasses: in banking transaction"
            )
            st.markdown(
                """### Transactions drivers used in prediction:
                    - step: represents a unit of time where 1 step equals 1 hour
                    - type: type of online transaction
                    - amount: the amount of the transaction
                    - nameOrig: customer starting the transaction
                    - oldbalanceOrg: balance before the transaction
                    - newbalanceOrig: balance after the transaction
                    - nameDest: recipient of the transaction
                    - oldbalanceDest: initial balance of recipient before the transaction
                    - newbalanceDest: the new balance of recipient after the transaction"""
            )

            # next = st.button("__Prediction__ ➡️")
            # if next:
            #     if st.session_state["current_page"] == "Main Page":
            #         st.session_state["current_page"] ="Prediction"

        elif page == "Prediction":
            st.title(":sunflower: :orange[Prediction]", anchor="Prediction")
            input_option = st.radio(
                ":blue[Pick data option]:lightning:", ["Single Record", "Upload File"]
            )
            if input_option == "Single Record":
                # Section for entering a single record
                st.subheader(":blue[Enter a record to predict]")
                column_list = model.get_frauddetection_input_schema()["columns"]
                single_record = {
                    column_list[0]: st.selectbox(
                        "Transacted type",
                        ("CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"),
                    ),
                    column_list[1]: st.number_input("Step number"),
                    column_list[2]: st.number_input("Amount"),
                    column_list[3]: st.number_input("OldBalance Origin"),
                    column_list[4]: st.number_input("NewBalance Origin"),
                    column_list[5]: st.number_input("Oldbalance Destination"),
                    column_list[6]: st.number_input("NewBalance Destination"),
                }

                single_record_df = pd.DataFrame([single_record])
                st.write(":blue[User Input (Single Record)]:")
                st.dataframe(single_record_df)

                if st.button(
                    "Predict :eyes:",
                    key="predict-button1",
                    type="primary",
                    help="Click to make predictions",
                ):
                    if not single_record_df.empty:
                        with st.spinner("Wait for it..."):
                            prediction_single = model.predict(single_record_df)
                        if not prediction_single.empty or prediction_single is not None:
                            st.success("This is a success", icon="✅")
                            st.snow()
                            st.write(
                                " 🉑 :red[Prediction for _Single Record_]: isFraud :blue[1=yes 0=Not fraud transaction]"
                            )
                            st.write(prediction_single)
                        else:
                            st.warning("This is a warning", icon="⚠️")

            if input_option == "Upload File":
                # Section for uploading a file
                with st.container():
                    st.subheader(
                        ":blue[Upload the data file to predict fraudulent transactions]"
                    )
                    uploaded_file = st.file_uploader(
                        "Choose a file", type=["csv", "xlsx"]
                    )
                    if uploaded_file is not None:
                        multiple_record_df = pd.read_csv(uploaded_file)
                        st.subheader(":blue[User Input DataFrame (Uploaded File)]:")
                        st.write(multiple_record_df)

                    # Perform prediction using the model
                    if st.button(
                        "Predict :eyes:",
                        key="predict-button",
                        type="primary",
                        help="Click to make predictions",
                    ):
                        if not multiple_record_df.empty:
                            with st.spinner("Wait for it..."):
                                prediction_multiple = model.predict(multiple_record_df)
                            if (
                                not prediction_multiple.empty
                                or prediction_multiple is not None
                            ):
                                st.success("This is a success", icon="✅")
                                st.balloons()
                                st.write(
                                    "🉑 :red[Prediction for _Multiple Records_]: isFraud :blue[1 = yes 0 = Not fraud transaction]"
                                )
                                st.write(prediction_multiple)
                            else:
                                st.warning("This is a warning", icon="⚠️")

        st.markdown(
            """<footer>  <p>Version: v1.0.0</p></footer>""", unsafe_allow_html=True
        )


if __name__ == "__main__":
    FraudDetectionPredictorApp.run()

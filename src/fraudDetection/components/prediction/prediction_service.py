import os
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import joblib
import json
import re
import streamlit as st

tags = {'Version':'v1'}
INPUT_SCHEMA_JSON_FILE_NAME = "input_schema.json"
MODEL_EVALUATION_YAML = "model_evaluation.yaml"
SAVED_MODELS_DIR = Path("artifacts\saved_models")
MODEL_NAME = "model.pkl"
PREPROCESSER_OBJ_PATH = Path(r"artifacts\transformed_data\preprocessed\preprocessed.pkl")
EVALUATED_MODELS_PATH = Path(r"artifacts\model_evaluation")

# schema_file_path = Path(r"C:\Users\arunk\FraudDetection\configs\schema.yaml")
class FraudDetectionModel():
    def __init__(self) -> None:
        self.model_name = MODEL_NAME
        self.evaluated_models_path = EVALUATED_MODELS_PATH 
        self.preprocessd_obj_path = PREPROCESSER_OBJ_PATH

    def get_latest_model(self):
        try:
            """Load best model from the saved models and the input schema"""
            self.evaluated_models_file_path = Path(os.path.join(self.evaluated_models_path,MODEL_EVALUATION_YAML))

            with open(self.evaluated_models_file_path, 'r') as f:
                model_registry_paths = yaml.load(f,Loader=yaml.Loader)
       
            # model_registry_paths = read_yaml(self.evaluated_models_file_path)
            path_object = model_registry_paths["best_model"]["model_path"]
            
            # Convert the WindowsPath object to a string
            path_string = str(path_object)
            # Define a regex pattern for matching the date
            date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}')
            # Search for the date pattern in the string
            match = date_pattern.search(path_string)
            # Extract the matched date
            if match:
                best_model_name_dir = match.group()

            matching_dirs = SAVED_MODELS_DIR.glob(best_model_name_dir)
            # Check if any matching directories were found
            if matching_dirs:
            # matching directory
                saved_model_path = first_string = next(matching_dirs, None)
                model_file = Path(os.path.join(saved_model_path,self.model_name))
                model = joblib.load(model_file)
        except (ValueError, TypeError) as e:
            print(f"Error: {e}")
        except Exception as e:
            print(e)
    
        return model
    
    def get_frauddetection_input_schema(self):
        self.input_schema_path = Path(os.path.join(self.evaluated_models_path, INPUT_SCHEMA_JSON_FILE_NAME))
        with open(self.input_schema_path, 'r') as json_file:
            schema = json.load(json_file)
        return schema
    
    def transform_input(self,df) -> np.array:
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
        try:
            model = self.get_latest_model()
            transformed_input = self.transform_input(df)
            prediction = model.predict(transformed_input)
            drop_features = ['isFlaggedFraud','isFraud']
            for col in drop_features:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            prediction_column_name = 'Prediction isFraud'
            df[prediction_column_name] = prediction
            
            return df
        except (ValueError, TypeError) as e:
            print(f"Error: {e}")
        except Exception as e:
            print(e)
        

class FraudDetectionPredictorApp:
    @staticmethod
    def run():
        st.set_page_config(
            page_title="Fraud Detection System",
            page_icon="🧊",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={'About': "# This is a Fraud Detection System! by Arun khare"}
            )
        st.caption(tags)
        model = FraudDetectionModel()

        page = st.sidebar.selectbox(
            """ Hello there! I’ll guide you! Please select Service""", 
            ["Main Page",
             "Prediction",
             "Training"
             ])
        
        if page == "Main Page":
            st.title("Hello, welcome to Fraud Detector")
            st.caption('This detects :blue[_Fraud Transaction_] :sunglasses: in banking transaction')
            st.markdown("""### Transactions drivers used in prediction:
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
           

        elif page == "Prediction":
            input_option = st.radio(":blue[Pick data option]:lightning:", ["Single Record", "Upload File"])
            if input_option == "Single Record":         
                # Section for entering a single record
                st.subheader(":blue[Enter a record to predict]")
                column_list = model.get_frauddetection_input_schema()['columns']            
                single_record = {    
                column_list[0]: st.selectbox('Transacted type', ('CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT')),
                column_list[1]: st.number_input('Step number'),
                column_list[2]: st.number_input('Amount'),
                column_list[3]: st.number_input('OldBalance Origin'),
                column_list[4]: st.number_input('NewBalance Origin'),
                column_list[5]: st.number_input('Oldbalance Destination'),
                column_list[6]: st.number_input('NewBalance Destination')
                }
     
                single_record_df = pd.DataFrame([single_record])
                st.write(":blue[User Input (Single Record)]:")
                st.dataframe(single_record_df)

                if st.button("Predict :eyes:", key="predict-button1",type="primary",help="Click to make predictions"):
                    if not single_record_df.empty:
                        with st.spinner('Wait for it...'):
                            prediction_single = model.predict(single_record_df)
                        if not prediction_single.empty or prediction_single is not None:
                            st.success('This is a success', icon="✅")
                            st.snow()
                            st.write("Prediction for Single Record Records: isFraud :blue[1=yes 0=Not a fraud transaction]")
                            st.write(prediction_single) 
                        else:
                            st.warning('This is a warning', icon="⚠️")

            if input_option == "Upload File":
                    # Section for uploading a file
                with st.container():
                    st.subheader(":blue[Upload the data file to predict fraudulent transactions]")
                    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
                    if uploaded_file is not None:
                        multiple_record_df = pd.read_csv(uploaded_file)
                        st.subheader(":blue[User Input DataFrame (Uploaded File)]:")
                        st.write(multiple_record_df)

                    #Perform prediction using the model
                    if st.button("Predict :eyes:", key="predict-button",type="primary",help="Click to make predictions"):
                        if not multiple_record_df.empty:
                            with st.spinner('Wait for it...'):
                                prediction_multiple = model.predict(multiple_record_df)
                            if not prediction_multiple.empty or prediction_multiple is not None:
                                st.success('This is a success', icon="✅")
                                st.balloons()
                                st.write("Prediction for Multiple Records: isFraud :blue[1=yes 0=Not a fraud transaction]")
                                st.write(prediction_multiple)
                            else: 
                                st.warning('This is a warning', icon="⚠️")

if __name__ == "__main__":
    FraudDetectionPredictorApp.run()

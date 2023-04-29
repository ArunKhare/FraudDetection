import os, sys
from box import ConfigBox
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from joblib import Memory
from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging
from fraudDetection.constants import *
from fraudDetection.entity import DataIngestionArtifact, DataTransformationConfig, DataValidationArtifact, DataTransformationArtifact
from fraudDetection.utils import save_object, read_yaml, load_data, save_numpy_array_data



class DataTransformation:
    def __init__(self, data_validation_artifact= DataValidationArtifact,
                 data_ingestion_artifact = DataIngestionArtifact,
                 data_transformation_config=DataTransformationConfig) -> None:
        try:
            logging.info(f"{'<<'*30}Data Transformation log started {'<<*30'}")
            self.data_validation_artifact = data_validation_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise FraudDetectionException(e,sys) from e 
    
    def get_data_transfomer_object(self):
        try:
            if not self.data_validation_artifact.is_validated:
                raise Exception ("Schema is not validated")
            # message = self.data_validation_artifact.message
            schema_file_path = self.data_validation_artifact.schema_file_path
            dataset_schema = read_yaml(schema_file_path)

            categorical_columns = dataset_schema[DATA_TRANSFORMATION_CATEGORICAL_COLUMN_KEY]
            numerical_columns = dataset_schema[DATA_TRANSFORMATION_NUMERICAL_COLUMN_KEY]
            
            num_pipeline = Pipeline(steps=[
            ('scaler', RobustScaler())
            ])
            cat_pipeline = Pipeline(steps= [
                ('ohe_type',OneHotEncoder())
            ])
            logging.info(f'Categorical Column: {categorical_columns}')            
            logging.info(f'Numerical columns : {numerical_columns}')

            transformer = ColumnTransformer([('cat_pipeline',cat_pipeline,categorical_columns),
                                             ('num_pipeline',num_pipeline,numerical_columns),
                                               ])
            model = ExtraTreesRegressor()
            preprocessing = Pipeline(steps=[('transformer',transformer),
                                       ("feature_extactor",model),])
            return preprocessing
        except Exception as e:
            raise FraudDetectionException(e,sys) from e 

    def initiate_data_transformation(self):
        try:
            logging.info(f"Obtaining preprocessing object")
            preprocessing_obj: Pipeline = self.get_data_transfomer_object()

            logging.info(f'obtaining training and test file path')
            train_file_path: Path = self.data_ingestion_artifact.train_file_path
            test_file_path: Path = self.data_ingestion_artifact.test_file_path

            schema_file_path: Path = self.data_validation_artifact.schema_file_path
            schema_file  = read_yaml (schema_file_path)
            schema=schema_file[DATA_VALIDATION_SCHEMA_COUMNS_KEY]

            logging.info(f'loading training and test data as pandas dataframe')
            train_df: pd.DataFrame = load_data(file_path=train_file_path,schema=schema)
            test_df:  pd.DataFrame = load_data(file_path=test_file_path,schema=schema)
     
            target_column_name = schema_file[DATA_TRANSFORMATION_TARGET_COLUMN_KEY][0]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
          

            logging.info(f'Applying preprocessing object on training and testing dataframe')

            # selecting features based on feature importance
            def get_selected_feature_names(pipe_obj,input_feature, target_feature) ->list:
                pipe_obj.fit(input_feature,target_feature)
                feature_out_names = pipe_obj[:-1].get_feature_names_out()
                feature_importance = pd.Series(data=pipe_obj[-1].feature_importances_, index=feature_out_names)
                top_10 = feature_importance.nlargest(10)
                return top_10[top_10 > 0].index.tolist()
            
            feature_selected = get_selected_feature_names(preprocessing_obj,input_feature_train_df,target_feature_train_df)
            logging.info(f'selected feature using feature importance \n {feature_selected} \n\n')                

            # get train and test dataframe from transformer Array output with column names
            def get_feature_df(pipe_obj,input_feature)  -> pd.DataFrame:          
                input_feature_arr = pipe_obj[:-1].fit_transform(input_feature)
                feature_out_names= pipe_obj[:-1].get_feature_names_out()
                feature_out_df = pd.DataFrame(data=input_feature_arr,columns=feature_out_names)
                return feature_out_df
            
            train_output_df = get_feature_df(preprocessing_obj, input_feature_train_df)
            selected_train_df = train_output_df[feature_selected]

            test_output_df = get_feature_df(preprocessing_obj, input_feature_test_df)
            selected_test_df = test_output_df[feature_selected]

            logging.info(f'train and test dataset columns \n {train_output_df.columns} \n {test_output_df.columns}')

            # concatenate the df + target column
            train_arr = np.c_[selected_train_df.to_numpy(), np.array(target_feature_train_df)]
            test_arr = np.c_[selected_test_df.to_numpy(), np.array(target_feature_test_df)]
           
            logging.info(f'\ntraining data array size: {train_arr.size} and shape: {train_arr.shape}')
            logging.info(f'\test data array size: {test_arr.size} and shape: {test_arr.shape}')

            # save the data transformed selected numpy array format dataset to artifacts path
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
            
            train_file_name = Path(os.listdir(train_file_path)[0].replace(".csv",".npy"))
            test_file_name = Path(os.listdir(test_file_path)[0].replace(".csv",".npy"))

            transformed_trained_file_path = Path(os.path.join(transformed_train_dir,train_file_name))
            transformed_test_file_path = Path(os.path.join(transformed_test_dir,test_file_name))

            logging.info(f"Saving transformed training and testing array at path \n {transformed_trained_file_path} \n {transformed_test_file_path}")
            chunk_size=10000
            # try:
            #     with open(file=transformed_trained_file_path, mode='ab') as f:
            #         for i in range(0, len(train_arr), chunk_size):
            #             np.save(f, train_arr[i:i+chunk_size])
            #             f.flush()
            #             os.fsync(f.fileno())
            # except Exception as e:
            #     raise FraudDetectionException(e,sys) from e
            print("Type of train_arr is : ============>" , type( train_arr))
            print("Transformed trained file path: =======>", transformed_trained_file_path)

            save_numpy_array_data(file_path=transformed_trained_file_path, array=train_arr, chunk_size=10000)
            save_numpy_array_data(file_path=transformed_test_file_path, array=test_arr, chunk_size=10000)

            # saving processing boject (pipe transformer object) to artifact path
            preprocessing_obj_file_path = self.data_transformation_config.preprocessing_object_dir
            save_object(preprocessing_obj_file_path,preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(
                is_transformed= True,
                message= "Transformed",
                transformed_train_file_path=transformed_trained_file_path,
                transformed_test_file_path=transformed_test_dir,
                processed_object_file_path=preprocessing_obj_file_path
            )
            logging.info(f'Data transformation artifact: {data_transformation_artifact}')
            return data_transformation_artifact
        except Exception as e:
            raise FraudDetectionException(e,sys) from e

    @staticmethod
    def handle_null_values(df):
        """
        Function to handle null values in a DataFrame by filling them with the median value of the column.
        args: dataframe
        return : dataframe
        """
        try:       
            null_values = df.isnull().sum().sum()
            if null_values > 0:
                df.fillna(df.median(), inplace=True)
                null_values = df.isnull().sum().sum()
                if null_values > 0:
                    raise Exception("DataFrame still contains null values.")
            return df
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30} Data transformation complete {'<<'*30} \n\n")
        


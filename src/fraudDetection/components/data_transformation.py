import os, sys
from box import ConfigBox
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin

from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging
from fraudDetection.constants import *
from fraudDetection.entity import DataIngestionArtifact, DataTransformationConfig, DataValidationArtifact, DataTransformationArtifact
from fraudDetection.utils import save_object, read_yaml, load_data, save_numpy_array_data
from sklearn import set_config
set_config(display='diagram')
from IPython.display import display


class SMOTENCWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features:list, sampling_strategy:float,target_feature_name:str):
        self.smotenc = SMOTENC(sampling_strategy=sampling_strategy, categorical_features=categorical_features)
        self.target_feature_name = target_feature_name

    def fit(self, X, y):
        self.X_resampled_, self.y_resampled_ = self.smotenc.fit_resample(X, y)
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        df_resampled: pd.DataFrame = pd.concat([pd.DataFrame(self.X_resampled_), pd.Series(self.y_resampled_, name=self.target_feature_name)], axis=1)
        return df_resampled
    
class RandomUnderSamplerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, sampling_strategy,target_feature_name):
        self.sampler = RandomUnderSampler(sampling_strategy=sampling_strategy)
        self.target_feature_name = target_feature_name
    def fit(self, X, y):
        self.X_resampled_, self.y_resampled_ = self.sampler.fit_resample(X.drop(columns=[self.target_feature_name]), X[self.target_feature_name])
        return self
    
    def transform(self, X, y=None) -> pd.DataFrame:
        df_resampled: pd.DataFrame = pd.concat([pd.DataFrame(self.X_resampled_), pd.Series(self.y_resampled_, name=self.target_feature_name)], axis=1)
        return df_resampled
    
class FeatureExtractor(BaseEstimator,TransformerMixin):
    def fit(self,X,y):
        self.model = ExtraTreesRegressor()
        self.model.fit(X,y)   
        return self
    def transform(self,X,y):
        return self

def get_data_balancing_object(target_feature_name) -> Pipeline:
    """
    define the preprocessor pipeline with SMOTENC and RandomUnderSampler
    Args: target feature column name
    return: pipeline object     
    """
    sampling_pipeline = Pipeline(steps=[
    ('sampling', SMOTENCWrapper(categorical_features=[1], sampling_strategy=0.1,target_feature_name=target_feature_name)),
    ('oversample', RandomUnderSamplerWrapper(sampling_strategy=0.5,target_feature_name=target_feature_name))
    ])
    return sampling_pipeline

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
            message = self.data_validation_artifact.message
            schema_file_path = self.data_validation_artifact.schema_file_path
            dataset_schema = read_yaml(schema_file_path)

            categorical_columns = dataset_schema[DATA_SCHEMA_CATEGORICAL_COLUMN_KEY]
            numerical_columns = dataset_schema[DATA_SCHEMA_NUMERICAL_COLUMN_KEY]
            
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
            train_file_path: Path = self.data_ingestion_artifact.train_file_path
            test_file_path: Path = self.data_ingestion_artifact.test_file_path
            logging.info(f'obtaining training and test file path: {train_file_path} \n {test_file_path}')

            schema_file_path: Path = self.data_validation_artifact.schema_file_path
            schema_file  = read_yaml (schema_file_path)
            schema=schema_file[DATA_SCHEMA_COLUMNS_KEY]
            target_feature_name = schema_file[DATA_SCHEMA_TARGET_COLUMN_KEY][0]

            train_df: pd.DataFrame = load_data(file_path=train_file_path,schema=schema)
            test_df:  pd.DataFrame = load_data(file_path=test_file_path,schema=schema)
            
            logging.info(f'\n loading training and test data: Shape {train_df.shape, test_df.shape}')

            DataTransformation.handle_null_values(train_df)
            DataTransformation.handle_null_values(test_df)
     
            input_feature_train_df = train_df.drop(columns=[target_feature_name],axis=1)
            target_feature_train_df = train_df[target_feature_name]

            input_feature_test_df = test_df.drop(columns=[target_feature_name],axis=1)
            target_feature_test_df = test_df[target_feature_name]
     
            logging.info(f'\n Split train DataFrame Input_feature_train_df {input_feature_test_df.shape} target_feature_train_df{target_feature_train_df.shape}')
            logging.info(f'\n Split test DataFrame Input_feature_train_df {input_feature_test_df.shape} target_feature_test_df {target_feature_test_df.shape}')
            
            logging.info(f"Obtaining processing objects")
            preprocessing_obj = self.get_data_transfomer_object()

            if self.data_validation_artifact.class_proportion_train > 0.10:
                #get over under sample from sample pipeline object
                sampling_obj: Pipeline = get_data_balancing_object(target_feature_name=target_feature_name)
                resampled_train_df = sampling_obj.fit_transform(input_feature_train_df,target_feature_train_df)
                resampled_X_train, resampled_y_train = resampled_train_df.drop(columns=[target_feature_name],axis=1), resampled_train_df[target_feature_name]

                # get scaled important features by feature selection 
                features_selected = self.get_selected_feature_names(preprocessing_obj,resampled_X_train,resampled_y_train)

                # get train and test dataframe from transformer Array output with column names
                train_output_df = self.get_feature_df(preprocessing_obj, resampled_X_train)
            else:
                
                features_selected = self.get_selected_feature_names(preprocessing_obj,input_feature_train_df,target_feature_train_df)
                              
                # get train and test dataframe from transformer Array output with column names
                train_output_df = self.get_feature_df(preprocessing_obj, input_feature_train_df)
            
            logging.info(f'\n selected feature using feature importance \n {features_selected} \n\n')  
            selected_train_df = train_output_df[features_selected]
        
            test_output_df = self.get_feature_df(preprocessing_obj, input_feature_test_df)
            selected_test_df = test_output_df[features_selected]

            logging.info(f'\n selected train and test dataset columns \n {train_output_df.columns} \n {test_output_df.columns}')
            logging.info(f'\n selected train and test dataframe shape {train_output_df.shape} {test_output_df.shape}')

            # concatenate the df + target column
            if not resampled_y_train.empty:
                train_arr = np.c_[selected_train_df.to_numpy(), np.array(resampled_y_train)]
            else:
                train_arr = np.c_[selected_train_df.to_numpy(), np.array(target_feature_train_df)]

            test_arr = np.c_[selected_test_df.to_numpy(), np.array(target_feature_test_df)]
           
            logging.info(f'\n training data array shape: {train_arr.shape}')
            logging.info(f'\n test data array shape: {test_arr.shape}')

            # save the data transformed selected numpy array format dataset to artifacts path
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
            
            train_file_name = os.listdir(train_file_path)[0].replace(".csv",".npz")
            test_file_name = os.listdir(test_file_path)[0].replace(".csv",".npz")
  
            transformed_trained_file_path = os.path.join(transformed_train_dir,train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir,test_file_name)

            logging.info(f"\n Saving transformed training and testing array at path {transformed_trained_file_path} \n {transformed_test_file_path}")
       
            save_numpy_array_data(file_path=transformed_trained_file_path, array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path, array=test_arr)

            # saving processing boject (pipe transformer object) to artifact path
            preprocessing_obj_file_path = Path(os.path.join(self.data_transformation_config.preprocessing_object_file_path))

            logging.info(f"Saving preprocessing obj at {preprocessing_obj_file_path}")
            transformer = preprocessing_obj.named_steps['transformer']
         
            save_object(preprocessing_obj_file_path, transformer)

            data_transformation_artifact = DataTransformationArtifact(
                is_transformed= True,
                message= "Transformed",
                transformed_train_file_path=transformed_trained_file_path,
                transformed_test_file_path=transformed_test_file_path,
                processed_object_file_path=preprocessing_obj_file_path
            )

            logging.info(f'Data transformation artifact: {data_transformation_artifact}')
            return data_transformation_artifact
        except Exception as e:
            raise FraudDetectionException(e,sys) from e

    @staticmethod
    # selecting features based on feature importance
    def get_selected_feature_names(pipe_obj, input_feature, target_feature) ->list:
        pipe_obj.fit(input_feature,target_feature)
        feature_out_names = pipe_obj[:-1].get_feature_names_out()
        feature_importance = pd.Series(data=pipe_obj[-1].feature_importances_, index=feature_out_names)
        top_10 = feature_importance.nlargest(10)
        return top_10[top_10 > 0].index.tolist()
            
    @staticmethod
    def get_feature_df(pipe_obj,input_feature) -> pd.DataFrame:          
        input_feature_arr = pipe_obj[:-1].fit_transform(input_feature)
        feature_out_names= pipe_obj[:-1].get_feature_names_out()
        feature_out_df = pd.DataFrame(data=input_feature_arr,columns=feature_out_names)
        return feature_out_df

    @staticmethod
    def handle_null_values(df):
        """
        Function to handle null values in a DataFrame by filling them with the median value of the column.
        - Drop columns with more than the upper threshold of null values
        - Drop rows with more than the upper threshold of null values
        - Fill remaining null values with columns medians
        - check if any null values remain in the DataFrame
        args: dataframe
        return : dataframe
        """
        try:
            null_prop =df.isnull().sum() /len(df)
            lower_thresh = 0.1
            upper_thresh = 0.3
          
            drops_cols = null_prop[null_prop > upper_thresh].index 
            df.drop(columns=drops_cols, inplace=True)
           
            drop_rows = df.loc[df.isnull().mean(axis=1).between(lower_thresh,upper_thresh)].index
            df.drop(index=drop_rows, inplace=True)
            
            df.fillna(df.median(),inplace=True)
          
            if df.isnull().sum().sum() >0:
                raise Exception("DataFrame still contains null values.")
        except Exception as e:
              raise FraudDetectionException(e,sys)
                
    def __del__(self):
        logging.info(f"{'>>'*30} Data transformation complete {'<<'*30} \n\n")
        
  # weights = {0: n_samples / (2 * np.bincount(y_train_c)[0]), 1: n_samples / (2 * np.bincount(y_train_c)[1])}


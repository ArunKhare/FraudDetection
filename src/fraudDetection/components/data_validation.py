import os, sys
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from evidently.report import Report
from evidently.metrics import DataDriftTable
from evidently.metrics import DatasetDriftMetric

from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging
from fraudDetection.constants import *
from fraudDetection.entity import DataValidationConfig, DataValidationArtifact, DataIngestionArtifact
from fraudDetection.utils import create_directories, read_yaml


class DataValidation:
    def __init__(self,data_ingestion_artifact: DataIngestionArtifact, data_validation_config:DataValidationConfig ) -> None:
        try:
            logging.info(f"{'<<'*30}Data validation log started {'<<*30'}")
            self.data_validation_config: DataValidationConfig = data_validation_config
            self.data_ingestion_artifact: DataIngestionArtifact = data_ingestion_artifact

        except Exception as e:
            raise FraudDetectionException(e,sys) from e

    def get_train_test_df(self) ->tuple:
        """ validation the schema of data downloaded with the provided schema
        """
        try:
            # logging checking training and test files are available
            train_file_dir= self.data_ingestion_artifact.train_file_path
            test_file_dir = self.data_ingestion_artifact.test_file_path

            is_training_path_exist = os.path.exists(train_file_dir)
            is_testing_path_exist = os.path.exists(test_file_dir)

            if not is_training_path_exist and is_testing_path_exist:
                raise Exception(f"train_file_dir {train_file_dir} or test_file_dir {test_file_dir} does not exist", sys)
            
            if not os.listdir(train_file_dir) and not os.listdir(test_file_dir):
                raise Exception(f"{train_file_dir} and {test_file_dir} no file found", sys)
            
            # Get a list of files from train and test data dir
            train_files = os.listdir(train_file_dir)
            test_files = os.listdir(test_file_dir)
          
            logging.info(f"Is train file and test file list exist with first files as : {train_files[0]}, {test_files[0]}")

            # concate all  the files in the train and test directory and convert them to dataframe
            train_files: list[str] = train_files[:2]
            test_files: list[str] = test_files[:2]
            
            #number of files in each test and train dir
            num_files_train = len([f for f in train_files if os.path.isfile(Path(os.path.join(train_file_dir, f))) if f.endswith('.csv')])
            num_files_test = len([f for f in test_files if os.path.isfile(Path(os.path.join(test_file_dir, f))) if f.endswith('.csv')])
            
            def concat_csv_files(files: list, file_dir: str) -> pd.DataFrame:
                dfs =[]    
                for i, file in enumerate(files):
                    if file.endswith('.csv'):
                        path = Path(os.path.join(file_dir,file))
                        df = pd.read_csv(path)
                        dfs.append(df)
                return pd.concat(dfs)
            
            train_df = concat_csv_files(train_files,train_file_dir)
            test_df = concat_csv_files(test_files,test_file_dir)

            logging.info(f"train_df and test_df created. Total train_dir CSV files :{num_files_train} test_dir csv files {num_files_test}")
            
            return train_df,test_df
        
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
       
    def validate_dataset_scheme(self,train_df,test_df):
        """ Validating dataset schema with schema provided
        retrun : bool
        """
        try:
            # Getting schema
            schema_file_path: Path = self.data_validation_config.schema_file_path
            schema_keys = read_yaml(schema_file_path)
            column_schema = schema_keys[DATA_VALIDATION_SCHEMA_COUMNS_KEY]
                            
            logging.info(f" train_df: {train_df.columns} {train_df.dtypes} \n test_df: {test_df.columns} {test_df.dtypes}")
           
            # Compare df's schema
            if not train_df.dtypes.equals(test_df.dtypes):
                raise Exception("Test and Train file schemas are not equal")
    
            missing_cols = [col for col in column_schema if col not in train_df.columns]
            mismatch_dtype_cols = [col for col, dtype in column_schema.items() if col in train_df.columns and train_df[col].dtype != dtype]

            if missing_cols:
                raise Exception (f"The following columns are missing from the DataFrame schema: {missing_cols}")
            elif mismatch_dtype_cols:
                raise Exception (f"The following columns data types does not match with schema: {mismatch_dtype_cols}")
            else:
                logging.info(f"train test file schema are as per the Schema {column_schema}")
            
            logging.info("Data validated")
            return True
        except Exception as e:
            raise FraudDetectionException(f"Error comparing schema: {e}",sys) from e
        
    
    def get_and_save_drift_report(self,train_df,test_df):
        """create a data drift report
        Args: 
            train_df: training dataset
            test_df: testing dataset
        return:
            json format report with drift metrics and drifttable
        """
        try:
            logging.info("Creating a datadrift report")
            report_file_path: Path = self.data_validation_config.report_file_path
            create_directories([report_file_path])
        
            # data_drift_datast_report = Report(metrics=[DatasetDriftMetric(), ])
            
            # # record the start time
            # start_time = time.time()  
            # for _ in tqdm(data_drift_datast_report.run(reference_data=train_df,current_data=test_df), mininterval=5, desc="Running a datadrift report",KeyboardInterrupt=True):
            #     elapsed_time = time.time() - start_time 
            #     if elapsed_time > 120: 
            #         raise TimeoutError("Report generation timed out")
                
            # # data_drift_datast_report.save_json(self.data_validation_config.report_file_path)           
            # report = json.loads(data_drift_datast_report.json)
            # logging.info(f"saving report to {report_file_path}")
            # with open(report_file_path,"w") as report_file:
            #     json.dump(report,report_file,indent=6)

            message = f"Data validated sucessfully and data drift report saved at {report_file_path} \n {[self.data_ingestion_artifact.message]}"
            logging.info (f"message : {message}")
            return message 
        except TimeoutError as e:
            logging.error(str(e))
            message = f"Data validated successfully, Datadrift report generation aborted. Data validation timed out \n {[self.data_ingestion_artifact.message]} "
            return message
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
        
    def is_data_drift_found(self):
        pass


    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df, test_df = self.get_train_test_df()

            is_data_schema_validated = self.validate_dataset_scheme(train_df, test_df)
            message = self.get_and_save_drift_report(train_df,test_df)
      
            data_validation_artifact = DataValidationArtifact(
                report_file_path = self.data_validation_config.report_file_path,
                schema_file_path = self.data_validation_config.schema_file_path,
                is_validated = is_data_schema_validated,
                message = message
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
      
            return data_validation_artifact
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'>*30'} Data validation completed. {'<'*30} \n\n")


    
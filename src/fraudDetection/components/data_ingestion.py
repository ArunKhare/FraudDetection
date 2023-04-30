import os,sys
import kaggle
from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
from joblib import Memory

from fraudDetection.entity import DataIngestionConfig, DataIngestionArtifact
from fraudDetection.logger import logging
from fraudDetection.exception import FraudDetectionException
from fraudDetection.utils import (
    create_directories,
    save_dfs_to_csv
)

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig) -> None:
        try:
            logging.info("{=*20} Data ingestion start {=*20}")
            self.data_ingestion_config: DataIngestionConfig = data_ingestion_config     
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
            
    def download_transaction_data(self,) -> str:
        try:
            #extraction remote url to download dataset
            download_dataset_link: str = self.data_ingestion_config.source_url

            #folder location to download file
            zip_download_dir: Path = self.data_ingestion_config.zip_dir
            base_filename = Path(os.path.basename(download_dataset_link.split()[-1]))
            zip_filepath:Path = Path(os.path.join(zip_download_dir, base_filename,) +".zip")
     
            create_directories([zip_download_dir])
          
            # Check if file already exists
            if os.path.exists(zip_filepath):
                logging.info(f"File {zip_filepath} already exists, skipping download")
                print("====>>>File already exists")
            else:
                logging.info(f"Downloading file from :[{download_dataset_link}] into :[{zip_download_dir}]")
                # Set the path to the kaggle.json file
            
                # Set the path to the kaggle.json file
                os.environ['KAGGLE_CONFIG_DIR'] = 'kaggle.json'
            
                dataset_name = Path(download_dataset_link.split()[-1])
                # Download the dataset using the Kaggle API
                tqdm(kaggle.api.dataset_download_files(dataset_name,zip_download_dir),mininterval=5,desc="downloading kaggle dataset in zip format" )

                logging.info(f"File :[{zip_download_dir}] has been downloaded successfully.")
            return zip_download_dir
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
    def unzipfile(self,zip_file_path) -> None:
        try:
            raw_data_dir: Path = self.data_ingestion_config.raw_data_dir
  
            create_directories([raw_data_dir])
            
            logging.info(f"Extracting zip file: [{zip_file_path}] into dir: [{raw_data_dir}]")
            file_name = Path(os.listdir(zip_file_path)[0])
               
            zip_file_path = Path(os.path.join(zip_file_path,file_name))
        
            size: int = os.path.getsize(zip_file_path)
            
            if size > 100:
                with ZipFile(file=zip_file_path, mode='r') as zip_ref:
                    zip_ref.extractall(path=raw_data_dir)
                logging.info(f"Extraction completed")
            else:
                raise FraudDetectionException("File is almost empty")
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            if not os.path.exists(self.data_ingestion_config.ingested_train_dir):
                is_ingested =False
                raw_data_dir: Path = self.data_ingestion_config.raw_data_dir
                if not os.listdir(raw_data_dir):
                    raise FraudDetectionException(f"No files found in the raw data directory: {raw_data_dir}", sys)
                if not os.listdir(raw_data_dir)[0]:
                    raise ValueError("Raw data directory is empty.")
                file_name = Path(os.listdir(raw_data_dir)[0])    
                file_path = Path(os.path.join(raw_data_dir, file_name))
                logging.info(f"Reading CSV file: {file_path}")
                df = pd.read_csv(file_path)

                # Check if test_size is within valid range
                if self.data_ingestion_config.test_size > 1 or self.data_ingestion_config.test_size <= 0:
                    raise FraudDetectionException(f"Invalid test_size: {self.data_ingestion_config.test_size}. Test size must be between 0 and 1", sys)

                # Check if stratify parameter refers to a valid column
                if self.data_ingestion_config.stratify not in df.columns:
                    raise FraudDetectionException(f"Invalid stratify column: {self.data_ingestion_config.stratify}. Column not found in DataFrame", sys)

                # Split the dataset into training and testing sets using stratified sampling
                logging.info("Split the dataset into training and testing sets using stratified sampling")
                strat_train_set, strat_test_set = train_test_split(df, test_size=self.data_ingestion_config.test_size, random_state=42, stratify=df[self.data_ingestion_config.stratify])

                # Export the training and testing datasets to files
                train_file_path = Path(os.path.join(self.data_ingestion_config.ingested_train_dir))
                test_file_path = Path(os.path.join(self.data_ingestion_config.ingested_test_dir))

                split_dataset_path: list[Path] = [train_file_path,test_file_path]

                if train_file_path is not None and test_file_path is not None:
                    create_directories(split_dataset_path)
                
                logging.info(f"Exporting training and testing dataset to files: [{train_file_path}, {test_file_path}] ")

                # Save each file as multiple chunks
                chunk_size = 100000
                tqdm(save_dfs_to_csv(strat_test_set, test_file_path, chunk_size),mininterval=5,desc=f"saving test data to csv {test_file_path}")
                tqdm(save_dfs_to_csv(strat_train_set, train_file_path, chunk_size),mininterval=5,desc= f"saving training dataset to csv{train_file_path}")
                message = self.check_class_balance(strat_train_set,strat_test_set,self.data_ingestion_config.stratify)
            else:         
                train_file_path = Path(os.path.join(self.data_ingestion_config.ingested_train_dir))
                test_file_path = Path(os.path.join(self.data_ingestion_config.ingested_test_dir))
                message = "Data already exist"  
            is_ingested=True
        
            # create the data ingestion artifacts
            data_ingestion_artifacts = DataIngestionArtifact(
                train_file_path=train_file_path,
                test_file_path=test_file_path,
                is_ingested=is_ingested,
                message=message
            )
            logging.info(f"Data ingestion artifacts : {data_ingestion_artifacts}")
         
            return data_ingestion_artifacts
        except ValueError as e:
            raise FraudDetectionException(str(e), sys) from e
        except FraudDetectionException as e:
            raise e
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            zip_file_path = self.download_transaction_data()
            self.unzipfile(zip_file_path)
            return self.split_data_as_train_test()
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
    @ staticmethod
    def check_class_balance(strat_train_set,strat_test_set,target_col) -> None:
        # compute class proportions in the training set
        train_counts = strat_train_set[target_col].value_counts()
        train_proportions = train_counts / len(strat_train_set)

        # compute class proportions in the test set
        test_counts = strat_test_set[target_col].value_counts()
        test_proportions = test_counts / len(strat_test_set)

        class_proportion_train = abs(train_proportions[0] - train_proportions[1])
        class_proportion_test = abs(test_proportions[1] - test_proportions[0])

        # check if the class proportions are proportional
        if class_proportion_train <= 0.05 and class_proportion_test <= 0.05:
            message: str = f'The class proportions are Balanced [Train : {class_proportion_train}, Test : {class_proportion_test}]' 
            logging.info(message)
        else:
            message:str = f'The class proportions are Balanced [Train : {class_proportion_train}, Test : {class_proportion_test}]' 
            logging.info(message)
        return message
        
    def __del__(self) -> None:
        logging.info(f"{'>>'*20} Data Ingestion log completed.{'<<'*20} \n\n")
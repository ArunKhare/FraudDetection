""" 
This module downloads the data and chunk, split the data in to train test
"""
import os
from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi
from fraudDetection.entity import DataIngestionConfig, DataIngestionArtifact
from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging
from fraudDetection.utils import (
    create_directories,
    save_dfs_to_csv,
    is_dir_empty,
    load_json,
    get_directory_size,
)


class DataIngestion:
    """
    Description:
        Downloaded datawith API. Process, split data,
    storing them to specified location
    Attributes:
        DataIngestionconfig: specfiying the artifacts paths
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig) -> None:
        try:
            logging.info(f"\n{'=' * 20} Data ingestion start {'=' * 20}")
            self.data_ingestion_config: DataIngestionConfig = data_ingestion_config
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def _authenticate_kaggle_api(self):
        """
        Authenticate Kaggle API using Kaggle.json
        return (str): api instance
        """
        conn_location = self.data_ingestion_config.kaggle_config_file
        connect = load_json(conn_location)
        # Kaggle
        os.environ["KAGGLE_CONFIG_DIR"] = str(conn_location)
        os.environ["KAGGLE_USERNAME"] = connect["username"]
        os.environ["KAGGLE_KEY"] = connect["key"]
        api = KaggleApi()
        api.authenticate()
        return api

    def download_transaction_data(
        self,
    ) -> str:
        """Downloading data from kaggle using API in directory"""

        try:
            # extraction remote url to download dataset
            download_dataset_link: str = self.data_ingestion_config.source_url
            raw_data_dir: Path = self.data_ingestion_config.raw_data_dir
            create_directories([raw_data_dir])

            logging.info(
                f"Downloading file from :[{download_dataset_link}] into :[{raw_data_dir}]"
            )
            assert isinstance(download_dataset_link, str) is True
            api = self._authenticate_kaggle_api()
            # Download the dataset using the Kaggle API
            tqdm(
                api.dataset_download_files(
                    download_dataset_link, raw_data_dir, unzip=True
                ),
                mininterval=5,
                desc="downloading kaggle dataset in zip format",
            )
            logging.info(f"File :[{raw_data_dir}] has been downloaded successfully.")

        except Exception as e:
            raise FraudDetectionException(e, error_details=sys) from e

    def split_data_as_train_test(self) -> DataIngestionArtifact:
        """
        Split the datase into train and test
        Returns:
            data_ingestion_artifacts(:obj:'DataIngestionArtifact'): Split datasets with message as DataIngestionartifact
        Example:
        data_ingestion_artifacts = create_data_ingestion_artifacts(
            train_file_path="/path/to/train.csv",
            test_file_path="/path/to/test.csv",
            is_ingested=True,
            message="Data successfully ingested."
        )
        logging.info(f"Data ingestion artifacts: {data_ingestion_artifacts}")
        """
        try:
            is_ingested = False
            raw_data_dir: Path = self.data_ingestion_config.raw_data_dir

            if is_dir_empty(raw_data_dir):
                raise ValueError("raw_data_dir must contain data file")

            directory_size = get_directory_size(raw_data_dir)

            logging.info(
                f"Total size of files in directory '{raw_data_dir}': {directory_size:0.2f} MB"
            )

            train_file_path = Path(
                os.path.join(self.data_ingestion_config.ingested_train_dir)
            )
            test_file_path = Path(
                os.path.join(self.data_ingestion_config.ingested_test_dir)
            )

            split_dataset_path: list[Path] = [train_file_path, test_file_path]
            create_directories(split_dataset_path)
            # check_data_dir(raw_data_dir)

            if is_dir_empty(train_file_path) or is_dir_empty(test_file_path):
                raise ValueError(
                    "Both train and test directories must contain data before splitting."
                )

            file_name = Path(os.listdir(raw_data_dir)[0])
            file_path = Path(os.path.join(raw_data_dir, file_name))
            logging.info(f"Reading CSV file: {file_path}")
            df = pd.read_csv(file_path)
            # Check if test_size is within valid range
            if (
                self.data_ingestion_config.test_size > 0.5
                or self.data_ingestion_config.test_size <= 0
            ):
                raise ValueError(
                    f"Invalid test_size: {self.data_ingestion_config.test_size}. Test size must be between 0 and 0.5",
                    sys,
                )
            # Check if stratify parameter refers to a valid column
            if self.data_ingestion_config.stratify not in df.columns:
                raise ValueError(
                    f"Invalid stratify column: {self.data_ingestion_config.stratify}. Column not found inDataFrame",
                    sys,
                )
            # Split the dataset into training and testing sets using stratified sampling
            logging.info(
                "Split the dataset into training and testing sets using stratified sampling"
            )
            strat_train_set, strat_test_set = train_test_split(
                df,
                test_size=self.data_ingestion_config.test_size,
                random_state=42,
                stratify=df[self.data_ingestion_config.stratify],
            )
            logging.info(
                f"Exporting training and testing dataset to files: [{train_file_path}, {test_file_path}] "
            )
            # Save each file as multiple chunks
            chunk_size = 100000
            save_dfs_to_csv(strat_test_set, test_file_path, chunk_size)
            save_dfs_to_csv(strat_train_set, train_file_path, chunk_size)
            # check_data_dir(test_file_path)
            # check_data_dir(train_file_path)
            message = f"Data ingestion completed successfully train and test data saved at {train_file_path}, {test_file_path}"
            is_ingested = True

            message = f"Train_test_split already exist at path {train_file_path} and {test_file_path}"

            # create the data ingestion artifacts
            data_ingestion_artifacts = DataIngestionArtifact(
                train_file_path=train_file_path,
                test_file_path=test_file_path,
                is_ingested=is_ingested,
                message=message,
            )
            logging.info(f"Data ingestion artifacts : {data_ingestion_artifacts}")
            return data_ingestion_artifacts

        except (ValueError, FraudDetectionException) as e:
            raise FraudDetectionException(e, sys) from e

    def user_input_to_downloaddata(self):
        """Choice to download  the data If 'y' download the data"""
        # Prompt the user for input
        print(
            f"{'*':.^10}Do you want to download transaction data? press y or n and enter"
        )

        if input().strip().lower() == "y":
            print("! Proceeding with Download ...")
            self.download_transaction_data()

        if input().strip().lower() == "n":
            print(
                f"\n{'_':_>10}You pressed 'n'. The data must be already downloaded {'_':_>10}\n"
            )
            logging.info("User pressed 'n',  data is not downloaded")

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Initiate data ingestion and Return the artifacts
        Returns:
            data_ingestion_artifacts(:obj:'DataIngestionArtifact')
        """
        try:
            self.user_input_to_downloaddata()
            return self.split_data_as_train_test()
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def __del__(self) -> None:
        logging.info(f"\n{'=' * 20} Data Ingestion Log Completed.{'=' * 20}\n\n")

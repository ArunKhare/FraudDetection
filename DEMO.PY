from fraudDetection.logger import logging
from fraudDetection.exception import FraudDetectionException
from fraudDetection.config.configuration import ConfigurationManager
from fraudDetection.components.data_ingestion import DataIngestion
from fraudDetection.constants import CONFIG_FILE_PATH
import os, sys

def main():
    try:
        config = ConfigurationManager(CONFIG_FILE_PATH)
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion_artifacts = DataIngestion(data_ingestion_config).initiate_data_ingestion()
        
    except Exception as e:
        logging.error(e)
        raise FraudDetectionException(e,sys) from e
    
if __name__ == "__main__":
    main()
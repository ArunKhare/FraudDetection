from fraudDetection.logger import logging
from fraudDetection.exception import FraudDetectionException
from fraudDetection.config.configuration import ConfigurationManager
from fraudDetection.components import DataIngestion, DataValidation, DataTransformation
from fraudDetection.entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from joblib import Parallel, delayed

from fraudDetection.constants import CONFIG_FILE_PATH
import os, sys

def main():
    try:
        config = ConfigurationManager(CONFIG_FILE_PATH)
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion_artifacts = DataIngestion(data_ingestion_config).initiate_data_ingestion()
        
        data_validation_config =config.get_data_validation_config()
        data_validation_artifacts =DataValidation(data_ingestion_artifacts,data_validation_config).initiate_data_validation()

        data_transformation_config =config.get_data_transformation_config()
        data_transformation_artifacts = DataTransformation(data_validation_artifacts,data_ingestion_artifacts,data_transformation_config).initiate_data_transformation()

    except Exception as e:
        logging.error(e)
        raise FraudDetectionException(e,sys) from e
    
if __name__ == "__main__":
    main()


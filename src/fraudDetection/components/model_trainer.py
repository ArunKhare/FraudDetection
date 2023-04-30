import os, sys
from typing import List

from fraudDetection.utils import create_directories, load_numpy_array_data, load_data, load_object
from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging
from fraudDetection.entity import ModelTrainerConfig, ModelTrainerArtifact, DataTransformationArtifact
from fraudDetection.utils import load_numpy_array_data, load_object, save_object

class FraudetectionEstimaorModel:
    def __init__(self) -> None:
        pass
    def predict(self,x)
        pass

class ModelTrainer:
    def __init__(self) -> None:
        pass
    def initiate_model_trainer(self):
        try:
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'>>'*30} Model trainer log completed {'<<'*30}")


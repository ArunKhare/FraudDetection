import os 
import urllib.request as request
from zipfile import ZipFile
from fraudDetection.entity import DataIngestionConfig
from fraudDetection.logger import logging
from fraudDetection.utils import get_size
from tqdm import tqdm
from pathlib import Path
import splitfolders


class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config
        

    def download_file(self):
        if not os.path.exists(self.config.raw_data_dir):
            logging.info("Download Started")
            try:
                
            except Exception as e:

        else:
            logging.info(
                f"file already exists of size: {get_size(Path(self.config.raw_data_dir))}"
            )

from fraudDetection.pipeline.pipeline import Pipeline
from fraudDetection.config.configuration import ConfigurationManager
from fraudDetection.constants import CONFIG_FILE_PATH

config = ConfigurationManager(config=CONFIG_FILE_PATH)

pipeline = Pipeline(config=config)

pipeline.run()

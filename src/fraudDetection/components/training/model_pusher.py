import os
import shutil
import sys

from fraudDetection.entity import ModelPusherConfig, ModelPusherArtifacts, ModelEvaluationArtifact
from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging
from fraudDetection.utils import create_directories
from pathlib import Path


class ModelPusher:

    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_evaluation_artifact: ModelEvaluationArtifact) -> None:

        try:

            logging.info(f"\n{'=' * 20} Model pusher log started {'=' * 20}")
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def export_model(self):
        try:

            evaluated_model_file_path = self.model_evaluation_artifact.evaluated_model_path
            is_model_accepted = self.model_evaluation_artifact.is_model_accepted

            export_dir = self.model_pusher_config.model_export_dir

            logging.info(f'evaluated_model_file_path : {evaluated_model_file_path}')

            if not is_model_accepted:
                response = ModelPusherArtifacts(
                    export_model_file_path=export_dir,
                    is_model_pusher=is_model_accepted
                )

                logging.info(response)

                return response

            model_file_name = os.path.basename(evaluated_model_file_path)
            export_model_file_path = Path(os.path.join(export_dir, model_file_name))

            create_directories([export_dir])

            shutil.copy(src=evaluated_model_file_path, dst=export_model_file_path)

            # we can call a function to save model to Azure blob storage/ google cloud storage / s3 bucket
            logging.info(f"Trained model: {evaluated_model_file_path} is copied in export dir: {export_dir}")

            model_pusher_artifact = ModelPusherArtifacts(
                export_model_file_path=export_model_file_path,
                is_model_pusher=is_model_accepted
                )

            logging.info(f"Model pusher artifact : {model_pusher_artifact}")

            return model_pusher_artifact

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def initiate_model_pusher(self):
        try:
            return self.export_model()
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def __del__(self):
        logging.info(f"\n {'=' * 20} Model Pusher log completed {'=' * 20} \n\n")

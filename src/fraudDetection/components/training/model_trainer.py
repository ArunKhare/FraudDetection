import sys
from pathlib import Path
from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging
from fraudDetection.entity import ModelTrainerConfig, ModelTrainerArtifact, DataTransformationArtifact
from fraudDetection.utils import load_numpy_array_data, save_object
from fraudDetection.entity import MetricInfoArtifact, ModelFactory
from fraudDetection.entity import evaluate_classification_model

import mlflow


class FraudetectionEstimaorModel:
    def __init__(self, trained_model_object) -> None:
        self.trained_model_object = trained_model_object

    def predict(self, X):
        return self.trained_model_object.predict(X)

    def __repr__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact) -> None:
        try:
            logging.info(f"\n{'=' * 20} Model training log started {'=' * 20}")
            self.model_trainer_config: ModelTrainerConfig = model_trainer_config
            self.data_transformation_artifact: DataTransformationArtifact = data_transformation_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        try:
            mlflow.start_run()
            # run = mlflow.active_run()
            # if not run.info.experiment_id:
            #     experiment_id = mlflow.create_experiment("experiment1")
                # mlflow.set_experiment(experiment_id=experiment_id)

            mlflow.set_tags({"version": "v1", "stage": "training"})

            is_trained = False

            logging.info(f"loading transformed training dataset")
            transformed_train_file_path = Path(self.data_transformation_artifact.transformed_train_file_path)
            train_array = load_numpy_array_data(transformed_train_file_path)

            logging.info(f"Loading transformed testing dataset")
            transformed_test_file_path = Path(self.data_transformation_artifact.transformed_test_file_path)
            test_array = load_numpy_array_data(transformed_test_file_path)

            logging.info(f"Splitting training and testing input and target feature")
            X_train, y_train, X_test, y_test = train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[
                                                                                                            :, -1]

            logging.info(f'Shapes: X_train {X_train}, y_train {y_train}, X_test {X_test}, y_test{y_test}')

            logging.info(f"Extracting model config file path")
            model_config_file_path = self.model_trainer_config.model_config_file_path

            logging.info(f"Initializing model factory class using above model config  file :{model_config_file_path}")

            model_factory = ModelFactory(model_config_path=model_config_file_path)
            base_score = self.model_trainer_config.base_score
            threshold = self.model_trainer_config.threshold_diff_train_test_acc

            logging.info(f"Expected  accuracy: {base_score}, threshold {threshold}")

            # mlflow.log_params({"base_score": base_score, "threshold": threshold})

            logging.info(f"Initiating operation model selection")
            best_model = model_factory.get_best_model(X=X_train, y=y_train, base_score=base_score)

            logging.info(f'Best Model found on training dataset: {best_model}')

            logging.info(f"Extracting trained model list")
            grid_searched_best_model_list = model_factory.grid_search_best_model_list

            model_list = [model.best_model for model in grid_searched_best_model_list]

            logging.info(f"Evaluation all trained model on training and testing dataset both")

            metric_info: MetricInfoArtifact = evaluate_classification_model(model_list=model_list, X_train=X_train,
                                                                            y_train=y_train, X_test=X_test,
                                                                            y_test=y_test, base_score=base_score,
                                                                            threshold=threshold)

            model_object = metric_info.model_object
            fraud_detection_model = FraudetectionEstimaorModel(trained_model_object=model_object)

            trained_model_file_path = Path(self.model_trainer_config.trained_model_file_path)
            logging.info(f"saving model at path: {trained_model_file_path}")

            if fraud_detection_model:
                logging.info(f"Best model found on both training and testing dataset")
                is_trained = True

            save_object(
                file_path=trained_model_file_path,
                obj=fraud_detection_model
            )

            model_trainer_artifact = ModelTrainerArtifact(
                is_trained=is_trained,
                message="Model Trained Successfully",
                trained_model_file_path=trained_model_file_path,
                train_accuracy=metric_info.train_accuracy_score,
                test_accuracy=metric_info.test_accuracy_score,
                train_f1_score=metric_info.train_f1_score,
                test_f1_score=metric_info.test_f1_score,
                train_fbeta_score=metric_info.train_fbeta_score,
                test_fbeta_score=metric_info.test_fbeta_score,
                train_roc_auc_score=metric_info.train_roc_auc_score,
                test_roc_auc_score=metric_info.test_roc_auc_score,
                train_precision_score=metric_info.train_precision_score,
                test_precision_score=metric_info.test_precision_score,
                train_recall_score=metric_info.train_recall_score,
                test_recall_score=metric_info.test_recall_score,
                model_accuracy=metric_info.model_accuracy,
                train_accuracy_score=metric_info.train_accuracy_score,
                test_accuracy_score=metric_info.test_accuracy_score,
                threshold=threshold
            )

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            mlflow_artifact_path = "trained_model"
            if  mlflow_artifact_path == mlflow.get_artifact_uri():
                mlflow.sklearn.save_model(sk_model=fraud_detection_model,
                                      path=mlflow_artifact_path,
                                      conda_env='conda.yaml',
                                      pyfunc_predict_fn="predict")

                mlflow.sklearn.log_model(sk_model=metric_info.model_object,
                                     artifact_path="trained_model",
                                     registered_model_name=metric_info.model_name,
                                     input_example=X_train[:1, :], metadata=dict(stage="trained",
                                                                                 index_number=metric_info.model_index))
            mlflow.log_params({"base_score": base_score, "threshold": threshold})
            mlflow.log_metric("train_f1_score", metric_info.train_f1_score)
            mlflow.log_metric("test_f1_score", metric_info.test_f1_score)
            mlflow.log_metric("train_fbeta_score", metric_info.train_fbeta_score)
            mlflow.log_metric("test_fbeta_score", metric_info.test_fbeta_score)
            mlflow.log_metric("train_roc_auc_score", metric_info.train_roc_auc_score)
            mlflow.log_metric("test_roc_auc_score", metric_info.test_roc_auc_score)
            mlflow.log_metric("train_precision_score", metric_info.train_precision_score)
            mlflow.log_metric("test_precision_score", metric_info.test_precision_score)
            mlflow.log_metric("train_recall_score", metric_info.train_recall_score)
            mlflow.log_metric("test_recall_score", metric_info.test_recall_score)
            mlflow.log_metric("model_accuracy", metric_info.model_accuracy)
            mlflow.log_metric("train_accuracy_score", metric_info.train_accuracy_score)
            mlflow.log_metric("test_accuracy_score", metric_info.test_accuracy_score)
            
            run =mlflow.active_run()
            if run:
                logging.info(f"Mlfow run is active {run.info}")

            return model_trainer_artifact

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def __del__(self):
        mlflow.end_run()
        logging.info(f"\n{'=' * 20} Model trainer log completed {'=' * 20} \n\n")

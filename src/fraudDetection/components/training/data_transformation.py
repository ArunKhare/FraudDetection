""" This module Transforms the validated data from earlier stage using 'Custom 
build classes'_ for SMOTENCWrapper, RandomUnderSamplerWrapper, CustomImputer"""
import os, sys
import numpy as np
import pandas as pd
from IPython.display import display
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile, chi2, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from pathlib import Path
from fraudDetection.constants import *
from fraudDetection.entity import (
    DataIngestionArtifact,
    DataTransformationConfig,
    DataValidationArtifact,
    DataTransformationArtifact,
)
from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging
from fraudDetection.utils import (
    save_object,
    read_yaml,
    load_data,
    save_numpy_array_data,
)

# set_config(display='diagram')


class SMOTENCWrapper(BaseEstimator, TransformerMixin):
    """Modified SMOTEC class for over sampling of data, takes Dataframe instead of numpy array
    Args:
        BaseEstimater (:obj:'sklearn.base'):base class that defines the basic interface for all estimators
        TransformerMixin (:obj:'sklearn.base'):adds and runs fit_transform without explicitly defining
    """

    def __init__(self, categorical_features: list, sampling_strategy: float):
        """
        Args:
            categorical_features ( list(str)): column list from schema
            sampling_strategy (float): desired ratio of the number of samples in the minority
                 class over the number of samples in the majority class after resampling
        """
        self.categorical_features = categorical_features
        self.sampling_strategy = sampling_strategy
        self.smotenc = SMOTENC(
            sampling_strategy=self.sampling_strategy,
            categorical_features=self.categorical_features,
        )

    def fit(self, X, y):
        """
        Takes training data and calculates the parameters and saves them as internal object state
        Args:
            x (pd.DataFrame): seq/  matrix of indexes of training data
            y (pd.Series): label or values to be predicted
        Returns:
            X_resampled (pd.Dataframe): referece to the internal object state
        """
        self.X_resampled_, self.y_resampled_ = self.smotenc.fit_resample(X, y)
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """
         use calculates the paramters in fit to transform the data
        Args:
            x (pd.DataFrame): seq/  matrix of indexes of training data
            y (pd.Series): label or values to be predicted
        Returns:
            df_resampled (pd.Dataframe) : generate synthetic samples for the minority class along with
                training data as output
        """

        self.df_resampled: pd.DataFrame = pd.concat(
            [self.X_resampled_, self.y_resampled_], axis=1
        )
        return self.df_resampled


class RandomUnderSamplerWrapper(BaseEstimator, TransformerMixin):
    """Custom class for under sampling of data
    Args:
        BaseEstimater (:obj:'sklearn.base'): base class that defines the basic interface for all estimators
        TransformerMixin (:obj:'sklearn.base'):adds fit_transform without explicitly defining
    """

    def __init__(self, sampling_strategy):
        """initialize the RandomunderSampler standard class
        Args:
            sampling_strategy (float): desired ratio of the number of samples in the minority
                 class over the number of samples in the majority class after resampling
        """
        self.sampling_strategy = sampling_strategy
        self.sampler = RandomUnderSampler(sampling_strategy=self.sampling_strategy)

    def fit(self, X, y):
        """
        Takes training data and calculates the parameters and saves them as internal object state
        Args:
            x (pd.DataFrame): seq/  matrix of indexes of training data
            y (pd.Series): label or values to be predicted
        Returns:
            X_resampled (pd.Dataframe): resampled data
        """
        self.X_resampled_, self.y_resampled_ = self.sampler.fit_resample(
            X.iloc[:, :-1], X.iloc[:, -1]
        )
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """Returns:
        X_resmapled (obj:'pd.DataFrame'): resample
        y_resampled (obj:'pd.DataFrame'): resample
        """
        return self.X_resampled_, self.y_resampled_


class CustomImputer(BaseEstimator, TransformerMixin):
    """Custom class for'Handling missing values'_ in data
    Args:
        BaseEstimater (:obj:'sklearn.base'): base class that defines the basic interface for all estimators
        TransformerMixin (:obj:'sklearn.base'):adds fit_transform without explicitly defining
    """

    def __init__(
        self, num_strategy="median", cat_strategy="most_frequent", thresh=None
    ):
        """initiallize the variables and standard SimpleImputer class object
        Args:
            num_strategy (str):  placeholder for the missing values in numerical feature
            cat_strategy (ste):  placeholder for the missing values in cateogrical feature
            thresh (float): percentage of missing values in the feature
            num_imputer(obj:'SimpleImputer'): obj of simple imputer for numerical features
            num_imputer(obj:'SimpleImputer'): obj of simple imputer for categorical features
        """
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.thresh = thresh
        self.num_imputer = SimpleImputer(strategy=self.num_strategy)
        self.cat_imputer = SimpleImputer(strategy=self.cat_strategy)

    def fit(self, X, y):
        """
        Takes training data and calculates the parameters and saves them as internal object state
        Args:
            x (pd.DataFrame): seq/  matrix of indexes of training data
            y (pd.Series): label or values to be predicted
        Returns:
            X_resampled (pd.Dataframe):fitted imputer
        """
        if self.thresh is not None:
            drop_cols = X.columns[X.isnull().mean() > self.thresh]
            X.drop(columns=drop_cols, inplace=True)

        num_cols = X.select_dtypes(include=np.number).columns
        self.num_imputer.fit(X[num_cols])

        cat_cols = X.select_dtypes(include="O").columns
        self.cat_imputer.fit(X[cat_cols])
        self.y = y
        self.X = X
        return self

    def transform(self, X, y=None):
        """
        Args:
            X (pd.DataFrame): X feature
        Returns:
            X (array): transformed arrayf
        """
        if self.thresh is not None:
            drop_cols = X.columns[X.isnull().mean() > self.thresh]
            X.drop(columns=drop_cols, inplace=True)

        num_cols = X.select_dtypes(include=np.number).columns
        X[num_cols] = self.num_imputer.transform(X[num_cols])

        cat_cols = X.select_dtypes(include="O").columns
        X[cat_cols] = self.cat_imputer.transform(X[cat_cols])

        return X


def get_data_processing_objects() -> Pipeline:
    """
    Module level function.
    Define the preprocessing data pipeline with SMOTENC and RandomUnderSampler
    Returns: pipe_obj(obj:'Pipeline'): Pipeline with fitted steps
             pipe_obj1(obj:'Pipeline'): Pipeline with fitted steps
    """
    imputer = Pipeline([("CustomImputer", CustomImputer(thresh=0.3))])

    sampling_pipeline = Pipeline(
        steps=[
            (
                "sampling",
                SMOTENCWrapper(categorical_features=[1], sampling_strategy=0.1),
            ),
            ("oversample", RandomUnderSamplerWrapper(sampling_strategy=0.5)),
        ]
    )

    cat_selector = make_column_selector(dtype_include=object)
    num_selector = make_column_selector(dtype_include=np.number)

    cat_preprocessor = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ("selector", SelectPercentile(chi2, percentile=50)),
        ]
    )

    num_preprocessor = Pipeline(steps=[("scaler", RobustScaler())])

    preprocessor = ColumnTransformer(
        [
            ("cat", cat_preprocessor, cat_selector),
            ("num", num_preprocessor, num_selector),
        ],
        remainder="passthrough",
    )

    feature_extractor = Pipeline(
        steps=[
            (
                "feature_selection",
                SelectFromModel(
                    ExtraTreesClassifier(n_estimators=50, max_features="log2"),
                    threshold=0.01,
                ),
            )
        ]
    )

    pipe_obj = Pipeline(
        [("CustomImputer", imputer), ("sampling_pipeline", sampling_pipeline)]
    )

    pipe_obj1 = Pipeline(
        [("transformer", preprocessor), ("FeatureExtractor", feature_extractor)]
    )

    return pipe_obj, pipe_obj1


def processing_data(
    impute_sampler_obj, preprocessor_obj, train_X, train_y, test_X, test_y
):
    """Transforms the data for 'Training and Testing'_
    Args:
        impute_sampler_obj  (obj:'Pipeline') : Pipeline steps for imputing missing values
        preprocessor_obj (obj:'Pipeline'): Pipline steps for processing categorical and numerical features
        train_X (obj:'pd.DataFrame'): Training features for training
        train_y (obj:'pd.DataFrame'): Training Target
        test_X (obj:'pd.DataFrame'): Testing feature
        test_y (obj:'pd.DataFrame'): Testing Target
    Returns:
        train_arr_with_y (ob:'Numpy.array'): training set Matrix
        test_arr_with_y (obj:'Numpy.array'): testing set Martrix
    """
    # preprocessing train dataset
    resample_train_X, resample_train_y = impute_sampler_obj.fit_transform(
        train_X, train_y
    )

    transformed_train_arr = preprocessor_obj.fit_transform(
        resample_train_X, resample_train_y
    )
    features_selected = preprocessor_obj.get_feature_names_out()

    logging.info(
        f"\n selected feature using feature importance \n {features_selected} \n\n"
    )

    # preprocessing test dataset
    if not test_X.isnull().sum().sum() == 0:
        test_X = impute_sampler_obj[-2:-1].fit_transform(test_X, test_y)

    transformed_test_arr = preprocessor_obj[-2:-1].fit_transform(test_X, test_y)
    test_feature_selected = preprocessor_obj[-2:-1].get_feature_names_out()

    if not transformed_train_arr.shape == transformed_test_arr.shape:
        test_df = pd.DataFrame(data=transformed_test_arr, columns=test_feature_selected)
        test_df_selected = test_df[features_selected]
        transformed_test_arr = np.array(test_df_selected)

    logging.info(
        f"\n selected train and test dataset shape \n {features_selected} \n {test_feature_selected}"
    )
    logging.info(
        f"\n selected train and test dataframe shape {transformed_train_arr.shape} {transformed_test_arr.shape}"
    )

    # concatenate the df + target column
    train_arr_with_y = np.c_[transformed_train_arr, np.array(resample_train_y)]
    test_arr_with_y = np.c_[transformed_test_arr, np.array(test_y)]
    logging.info(f"\n training data array with y shape: {train_arr_with_y.shape}")
    logging.info(f"\n test data array with y shape: {test_arr_with_y.shape}")

    return train_arr_with_y, test_arr_with_y


class DataTransformation:
    """implementing transformation of data
    Args:
    data_validation_artifact (obj:'DataValidationArtifact'): validated data artifacts
    data_ingestion_artifact (obj:'DataIngestionArtifact'): data ingested artifacts
    data_transformation_config (obj:'DataTransformationConfig'): data transformation artifacts
    """

    def __init__(
        self,
        data_validation_artifact=DataValidationArtifact,
        data_ingestion_artifact=DataIngestionArtifact,
        data_transformation_config=DataTransformationConfig,
    ) -> None:
        """initializing artifacts classes"""
        try:
            logging.info(f"\n{'=' * 20}Data Transformation log started {'=' * 20}")
            self.data_validation_artifact = data_validation_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def initiate_data_transformation(self):
        """initialize and implement the data transformation returns the path to array serialized data files
        Args:
            train_file_path (obj:'pathlib.Path'):directory path to csv training dataset
            test_file_path (obj:'pathlib.Path'):directory path to csv test dataset
        Returns:
         data_transformation_artifact (obj:'DataTransformationArtifact'): transformed artifacts

        """
        try:
            train_file_path: Path = self.data_ingestion_artifact.train_file_path
            test_file_path: Path = self.data_ingestion_artifact.test_file_path
            logging.info(
                f"obtaining training and test file path: {train_file_path} \n {test_file_path}"
            )

            schema_file_path: Path = self.data_validation_artifact.schema_file_path
            schema = read_yaml(schema_file_path)
            data_schema = schema[DATA_SCHEMA_COLUMNS_KEY]
            target_feature_name = schema[DATA_SCHEMA_TARGET_COLUMN_KEY][0]

            train_df: pd.DataFrame = load_data(
                file_path=train_file_path, schema=data_schema, args=(None, None)
            )
            test_df: pd.DataFrame = load_data(
                file_path=test_file_path, schema=data_schema, args=(None, 50)
            )

            logging.info(
                f"\n loading training and test data: Shape {train_df.shape, test_df.shape}"
            )

            input_feature_train_df = train_df.drop(
                columns=[target_feature_name], axis=1
            )
            target_feature_train_df = train_df[target_feature_name]

            input_feature_test_df = test_df.drop(columns=[target_feature_name], axis=1)
            target_feature_test_df = test_df[target_feature_name]

            logging.info(
                f"\n Split train DataFrame Input_feature_train_df {input_feature_train_df.shape} target_feature_train_df{target_feature_train_df.shape}"
            )

            logging.info(
                f"\n Split test DataFrame Input_feature_train_df {input_feature_test_df.shape} target_feature_test_df {target_feature_test_df.shape}"
            )

            inputer_sampler_obj, preprocessor_obj = get_data_processing_objects()

            logging.info(f"processing pipeline : {display(preprocessor_obj)}")

            train_arr_with_y, test_arr_with_y = processing_data(
                inputer_sampler_obj,
                preprocessor_obj,
                input_feature_train_df,
                target_feature_train_df,
                input_feature_test_df,
                target_feature_test_df,
            )

            # save the data transformed selected numpy array format dataset to artifacts path
            transformed_train_dir = (
                self.data_transformation_config.transformed_train_dir
            )
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.listdir(train_file_path)[0].replace(".csv", ".npz")
            test_file_name = os.listdir(test_file_path)[0].replace(".csv", ".npz")

            transformed_trained_file_path = os.path.join(
                transformed_train_dir, train_file_name
            )
            transformed_test_file_path = os.path.join(
                transformed_test_dir, test_file_name
            )

            logging.info(
                f"\n Saving transformed training and testing array at path {transformed_trained_file_path} \n {transformed_test_file_path}"
            )

            save_numpy_array_data(
                file_path=transformed_trained_file_path, array=train_arr_with_y
            )
            save_numpy_array_data(
                file_path=transformed_test_file_path, array=test_arr_with_y
            )

            # saving processing boject (pipe transformer object) to artifact path
            preprocessing_obj_file_path = Path(
                os.path.join(
                    self.data_transformation_config.preprocessing_object_file_path
                )
            )
            impute_sample_obj_file_path = Path(
                os.path.join(
                    self.data_transformation_config.imputer_sampler_object_file_path
                )
            )

            logging.info(f"Saving preprocessing obj at {preprocessing_obj_file_path}")

            save_object(preprocessing_obj_file_path, preprocessor_obj)
            save_object(impute_sample_obj_file_path, inputer_sampler_obj)

            data_transformation_artifact = DataTransformationArtifact(
                is_transformed=True,
                message="Transformed",
                transformed_train_file_path=transformed_trained_file_path,
                transformed_test_file_path=transformed_test_file_path,
                processed_object_file_path=preprocessing_obj_file_path,
                impute_sampler_object_file_path=impute_sample_obj_file_path,
            )

            logging.info(
                f"Data transformation artifact: {data_transformation_artifact}"
            )
            return data_transformation_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def __del__(self):
        logging.info(f"\n{'=' * 20} Data Transformation Log Complete {'=' * 20}\n\n")

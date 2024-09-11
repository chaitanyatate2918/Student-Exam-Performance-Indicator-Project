import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Numerical Pipeline is created")

            cat_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("categprical pipeeline is created")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ],
                remainder='passthrough'
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        

    def initate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data transformation is started")
            logging.info("Reading training and testing data")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Obtaining preprocessor object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column = "math_score"
            drop_column = [target_column]

            input_feature_train_data = train_data.drop([target_column], axis=1)
            target_feature_train_data = train_data[target_column]

            input_feature_test_data = test_data.drop([target_column], axis=1)
            target_feature_test_data = test_data[target_column]

            input_train_arr = preprocessing_obj.fit_transform(input_feature_train_data)
            input_test_arr = preprocessing_obj.transform(input_feature_test_data)

            logging.info("Apply preprocessor object on our train data and test data")
            train_array = np.c_[input_train_arr, np.array(target_feature_train_data)]
            test_array = np.c_[input_test_arr, np.array(target_feature_test_data)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Saved preprocessor object")
            logging.info("Data transformation process is completed")

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path,
            ) 
        
        except Exception as e:
            raise CustomException(e,sys)



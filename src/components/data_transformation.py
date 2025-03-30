import os
import sys
import pickle
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# Data Transformation Configuration
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts/preprocessor.pkl')


# Data Transformation Class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Data Categorization
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
           
            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info(f"Numerical columns standard scaling completed: {numerical_columns}")
            logging.info(f"Categorical columns one-hot encoding completed: {categorical_columns}")

            # Column Transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            logging.info("Preprocessor object created successfully.")
            return preprocessor
        
        except Exception as e:
            logging.error(f"Error loading preprocessor object: {e}")
            raise CustomException(e, sys) from e


    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load Data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Data loaded successfully.")
            logging.info(f"Train DataFrame shape: {train_df.shape}")
            logging.info(f"Test DataFrame shape: {test_df.shape}")


            logging.info(f"Obtaining preprocessor object.")
            preprocessor_obj = self.get_data_transformer_object()
            logging.info("Preprocessor object obtained successfully.")

            # Apply Preprocessing
            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']

            #Train Data
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Train Data
            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Preprocessing completed successfully for training and Testing data.")



            #Fitting training data to preprocessor object
            input_features_train_array = preprocessor_obj.fit_transform(input_features_train_df)


            #Transform test data to preprocessor object
            input_features_test_array = preprocessor_obj.transform(input_features_test_df)
            logging.info("Preprocessing completed successfully for training and Testing data.")

            # Convert to Numpy Array
            train_arr = np.c_[
                input_features_train_array, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_features_test_array, np.array(target_feature_test_df)
            ]
            logging.info("Data transformation completed successfully.")

            # Save Transformed Data

            logging.info("Saving preprocessor object.")
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )


            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise CustomException(e, sys) from e
        



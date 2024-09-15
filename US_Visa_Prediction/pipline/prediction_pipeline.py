import os
import sys
import numpy as np
import pandas as pd
from US_Visa_Prediction.entity.config_entity import USvisaPredictorConfig
from US_Visa_Prediction.exception import USvisaException
from US_Visa_Prediction.logger import logging
from US_Visa_Prediction.utils.main_utils import read_yaml_file
from pandas import DataFrame
from US_Visa_Prediction.entity.estimator import USvisaModel  # Assuming this loads local models

class USvisaData:
    def __init__(self,
                 continent,
                 education_of_employee,
                 has_job_experience,
                 requires_job_training,
                 no_of_employees,
                 region_of_employment,
                 prevailing_wage,
                 unit_of_wage,
                 full_time_position,
                 company_age):
        """
        USvisa Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.company_age = company_age

        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_usvisa_input_data_frame(self) -> DataFrame:
        """
        This function returns a DataFrame from USvisaData class input
        """
        try:
            usvisa_input_dict = self.get_usvisa_data_as_dict()
            return DataFrame(usvisa_input_dict)
        
        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_usvisa_data_as_dict(self):
        """
        This function returns a dictionary from USvisaData class input 
        """
        logging.info("Entered get_usvisa_data_as_dict method as USvisaData class")

        try:
            input_data = {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "company_age": [self.company_age],
            }

            logging.info("Created USvisa data dict")

            logging.info("Exited get_usvisa_data_as_dict method as USvisaData class")

            return input_data

        except Exception as e:
            raise USvisaException(e, sys) from e
        

import pickle
import os
import sys
from US_Visa_Prediction.exception import USvisaException
from US_Visa_Prediction.logger import logging

class USvisaModel:
    @staticmethod
    def load(file_path: str):
        """
        Load the saved model from the given file path.
        """
        try:
            logging.info(f"Loading model from {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Model file not found at {file_path}")
                
            with open(file_path, 'rb') as file:
                model = pickle.load(file)
                
            logging.info(f"Model loaded successfully from {file_path}")
            return model
        except Exception as e:
            raise USvisaException(f"Error loading model from {file_path}: {str(e)}", sys)








class USvisaClassifier:
    def __init__(self, prediction_pipeline_config: USvisaPredictorConfig = USvisaPredictorConfig()) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise USvisaException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This method predicts the outcome using the locally loaded model
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of USvisaClassifier class")
            model_path = self.prediction_pipeline_config.model_file_path  # Use local path instead of S3
            logging.info(f"Loading model from path: {model_path}")

            if os.path.exists(model_path):
                model = USvisaModel.load(model_path)
                result = model.predict(dataframe)
            else:
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            logging.info("Exited predict method of USvisaClassifier class with success")
            return result
        
        except Exception as e:
            raise USvisaException(e, sys)


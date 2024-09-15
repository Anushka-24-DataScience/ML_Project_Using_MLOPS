import os
from US_Visa_Prediction.entity.config_entity import ModelEvaluationConfig
from US_Visa_Prediction.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from US_Visa_Prediction.exception import USvisaException
from US_Visa_Prediction.constants import TARGET_COLUMN, CURRENT_YEAR
from US_Visa_Prediction.logger import logging
import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from US_Visa_Prediction.entity.estimator import USvisaModel, TargetValueMapping
from US_Visa_Prediction.utils.main_utils import load_object  

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_best_model(self) -> Optional[USvisaModel]:
        """
        Method Name :   get_best_model
        Description :   This function loads the best model from the local path
        
        Output      :   Returns model object if available in the local storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            model_path = self.model_eval_config.path_name  # Using local path instead of S3
            logging.info(f"Loading best model from path: {model_path}")
            
            if model_path and os.path.exists(model_path):
                best_model = load_object(file_path=model_path)
                return best_model
            return None
        except Exception as e:
            raise USvisaException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function evaluates the trained model against the best model (if exists)
        
        Output      :   Returns an evaluation response with model comparison
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df['company_age'] = CURRENT_YEAR - test_df['yr_of_estab']

            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            y = y.replace(TargetValueMapping()._asdict())

            # Evaluate the trained model
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            # Check if there's an existing best model to compare
            best_model_f1_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
            
            # Calculate the difference between the trained and best model
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=tmp_best_model_score,
                is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                difference=trained_model_f1_score - tmp_best_model_score
            )
            logging.info(f"Evaluation result: {result}")
            return result

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function initiates the model evaluation process
        
        Output      :   Returns a ModelEvaluationArtifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            evaluate_model_response = self.evaluate_model()

            # Create a model evaluation artifact without referencing S3
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                path_name=self.model_eval_config.path_name,  # Local path to the best model
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise USvisaException(e, sys)

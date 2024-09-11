import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException 
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



# Decorator (after defining dataclass no need to create constuctor for class variables)
@dataclass
# It pass as the input to DataIngestion method and specify path to store the output files
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts/data_ingestion", "train.csv")
    test_data_path:str = os.path.join("artifacts/data_ingestion", "test.csv")
    raw_data_path:str = os.path.join("artifacts/data_ingestion", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Initiate the data ingestion process")
            logging.info("Data Reading using Pandas library from local system")

            data=pd.read_csv(os.path.join("notebook/data", "stud.csv"))
            logging.info("Data reading completed")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("initiate train test split")
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data Ingestion process is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)


# if __name__ == "__main__":
#     obj = DataIngestion()
#     train_data, test_data = obj.initiate_data_ingestion()

#     data_transform = DataTransformation()
#     train_array, test_array,_ = data_transform.initate_data_transformation(train_data, test_data)

#     modeltrainer = ModelTrainer()
#     print(modeltrainer.initiate_model_trainer(train_array, test_array))




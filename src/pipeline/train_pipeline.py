from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path =  obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array,_ = data_transformation.initate_data_transformation(train_data_path,test_data_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_array, test_array)

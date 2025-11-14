import mlflow
import dagshub
import json
import pandas as pd
import joblib
from pathlib import Path
import logging
import traceback
from sklearn import set_config
from sklearn.metrics import mean_absolute_percentage_error


dagshub.init(repo_owner='pankajireo74', repo_name='uber-demand-prediction', mlflow=True)

# set the mlflow tracking uri
mlflow.set_tracking_uri("https://dagshub.com/pankajireo74/uber-demand-prediction.mlflow")

# set the experiment name
mlflow.set_experiment("DVC Pipeline")

set_config(transform_output="pandas")

# create a logger
logger = logging.getLogger("evaluate_model")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

def load_model(model_path):
    model = joblib.load(model_path)
    return model


def save_run_information(run_id, artifact_path, model_uri, path):
    run_information = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_uri": model_uri
    }
    with open(path, "w") as f:
        json.dump(run_information, f, indent=4)


if __name__ == "__main__":
    # current path
    current_path = Path(__file__)
    # set the root path
    root_path = current_path.parent.parent.parent
    # data_path
    train_data_path = root_path / "data/processed/train.csv"
    test_data_path = root_path / "data/processed/test.csv"
    
    # read the data
    df = pd.read_csv(test_data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("Data read successfully")
    
    # set the datetime column as index
    df.set_index("tpep_pickup_datetime", inplace=True)
    
    # make X_test and y_test
    X_test = df.drop(columns=["total_pickups"])
    y_test = df["total_pickups"]
    
    # load the encoder
    encoder_path = root_path / "models/encoder.joblib"
    encoder = joblib.load(encoder_path)
    logger.info("Encoder loaded successfully")
    
    # transform the test data
    X_test_encoded = encoder.transform(X_test)
    logger.info("Data transformed successfully")
    
    # load the model
    model_path = root_path / "models/model.joblib"
    model = load_model(model_path)
    logger.info("Model loaded successfully")
    
    # make predictions
    y_pred = model.predict(X_test_encoded)
    
    # calculate the loss
    loss = mean_absolute_percentage_error(y_test, y_pred)
    logger.info(f"Loss: {loss}")
    
    # mlflow tracking
    try:
        with mlflow.start_run(run_name="model") as run:    
            # log the model parameters
            mlflow.log_params(model.get_params())
            
            # log the mertic
            mlflow.log_metric("MAPE", loss)
            
            # converts the datasets into mlfow datasets
            training_data = mlflow.data.from_pandas(pd.read_csv(train_data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime"), targets="total_pickups")
            
            validation_data = mlflow.data.from_pandas(pd.read_csv(test_data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime"), targets="total_pickups")
            
            # log the datasets
            mlflow.log_input(training_data, "training")
            mlflow.log_input(validation_data, "validation")
            
            # model signature
            model_signature = mlflow.models.infer_signature(X_test_encoded, y_pred)
            
            # get absolute path for requirements.txt
            requirements_path = root_path / "requirements.txt"
            
            # get the run id first
            run_id = run.info.run_id
            artifact_path = "demand_prediction"
            
            # log sklearn model
            # Note: Using MLflow 2.7.0 for DagsHub compatibility (MLflow 3.x not supported yet)
            try:
                logged_model = mlflow.sklearn.log_model(
                    model, 
                    artifact_path, 
                    signature=model_signature,
                    pip_requirements=str(requirements_path) if requirements_path.exists() else None
                )
                model_uri = logged_model.model_uri
                logger.info("Model logged successfully")
            except Exception as e:
                # Fallback: try without pip_requirements if there's an error
                logger.warning(f"Error logging model with pip_requirements: {e}. Trying without pip_requirements.")
                logged_model = mlflow.sklearn.log_model(
                    model, 
                    artifact_path, 
                    signature=model_signature
                )
                model_uri = logged_model.model_uri
                logger.info("Model logged successfully without pip_requirements")
            
            logger.info("Mlflow logging complete")
            
            # save to json file
            json_file_save_path = root_path / "run_information.json"
            save_run_information(run_id=run_id,
                                 artifact_path=artifact_path,
                                 model_uri=model_uri,
                                 path=json_file_save_path)
            logger.info("Run information saved successfully")
            
    except Exception as e:
        logger.error(f"Error during MLflow logging: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error("Note: If this is an authentication error, ensure DagsHub credentials are configured.")
        logger.error("You may need to set DAGSHUB_USER_TOKEN environment variable or run: dagshub.auth.add_app_token()")
        raise
    
    
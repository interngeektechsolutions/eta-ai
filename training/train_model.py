import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import argparse
import os
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(data_path:str="data/training_data.csv",model_path:str="models/eta_model.pkl",n_estimators:int=100):
    logger.info(f"Loading training data from '{data_path}'")
    data = pd.read_csv(data_path)
    
    X = data[["distance","speed","hour","weekday"]]
    Y = data["eta"]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    logger.info(f"Training RandomForestRegressor with {n_estimators} estimators")
    model = RandomForestRegressor(n_estimators=n_estimators,random_state=42,n_jobs=-1)
    model.fit(X_train,Y_train)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(Y_test,predictions)
    r2 = r2_score(Y_test,predictions)
    logger.info(f"Evaluation - MAE: {mae:.4f}, R2 Score: {r2:.4f}")
    
    os.makedirs(os.path.dirname(model_path),exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to '{model_path}'")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ETA prediction model") 
    parser.add_argument("--data",type=str,default="data/training_data.csv",help="Path to training data CSV file")
    parser.add_argument("--model",type=str,default="models/eta_model.pkl",help="Output model path")
    parser.add_argument("--estimators",type=int,default=100,help="Number of estimators")
    args = parser.parse_args()
    train(data_path=args.data, model_path=args.model, n_estimators=args.estimators) 
    
     
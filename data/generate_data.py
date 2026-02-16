import pandas as pd 
import numpy as np
import argparse
import os

def generate_data(rows:int=5000,output_path:str="data/training_data.csv"):
    np.random.seed(42)
    data=[] 

    for _ in range(rows):
        distance = np.random.uniform(0.5,10)
        speed = np.random.uniform(10,40)
        hour = np.random.randint(0,24)
        weekday = np.random.ranint(0,7)
        
        traffic_factor = 1.5 if 8<= hour <= 10 or 17 <= hour <=19 else 1.0
        eta = (distance/speed) * traffic_factor 
        data.append([distance,speed,hour,weekday,eta])
    df = pd.DataFrame(data,columns=["distance","speed","hour","weekday","eta"])
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    df.to_csv(output_path,index=False)
    print(f"Generated {rows} rows of training data at '{output_path}'") 
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training data for ETA prediction")
    parser.add_argument("--rows",type=int,default=5000,help="Number of data rows to generate")
    parser.add_argument("--output",type=str,default="data/training_data.csv",help="Path to save the generated CSV file")
    args = parser.parse_args()
    generate_data(rows=args.rows,output_path=args.output)                    
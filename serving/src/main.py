import os
import pickle
import pandas as pd
from fastapi import FastAPI, Body
from typing import List, Dict, Any
from pydantic import BaseModel

with open("pipeline.pkl", "rb") as f:
        model = pickle.load(f)

class Dataset(BaseModel):
    data: List

app = FastAPI()

@app.post("/predict")
def get_prediction(dataset: Dict[str, List[Dict[str, Any]]]):
    
    data = dataset['data']
    # data = dataset.data
    data = pd.DataFrame(data)

    prediction = model.predict(data)
    prediction.to_dict('records')
    return prediction
    # prediction_list = prediction.tolist()
    # return {"predictions": prediction_list}


if __name__ == "__main__":
    print("test")

import os
import pickle

from library import *
from pydantic import BaseModel, conlist
from typing import List
from fastapi import FastAPI, Body

with open("pipeline.pkl", "rb") as f:
        model = pickle.load(f)

class Dataset(BaseModel):
    data: List

app = FastAPI()

@app.post("/predict")
def get_prediction(dataset: Dataset):
    
    data = dataset['data']
    data = pd.DataFrame(data)

    prediction = model.predict(data)
    prediction.to_dict('records')
    return prediction


if __name__ == "__main__":
    print("test")

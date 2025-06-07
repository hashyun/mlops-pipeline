import pandas as pd
from pathlib import Path

from src.modules.lstm_utils import (
    load_data,
    train_lstm,
    save_model,
    generate_predictions
)

TRAIN_PATH = 'notebook/data/merged_train (1).csv'
TEST_PATH = 'notebook/data/merged_test (1).csv'
MODEL_PATH = 'notebook/weights/lstm_model.pth'
PRED_PATH = 'notebook/data/predictions.csv'


def lstm_train_task():
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    model = train_lstm(train_df)
    Path('notebook/weights').mkdir(parents=True, exist_ok=True)
    save_model(model, MODEL_PATH)


def prediction_task():
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    model = train_lstm(train_df)
    preds = generate_predictions(model, train_df, test_df)
    result = pd.DataFrame({'date': test_df['date'], 'actual_close': test_df['Close'], 'pred_close': preds})
    result.to_csv(PRED_PATH, index=False)


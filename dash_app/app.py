import pandas as pd
from dash import Dash, html, dcc
import plotly.graph_objs as go
from src.modules.lstm_utils import load_model, generate_predictions

TRAIN_PATH = 'notebook/data/merged_train (1).csv'
TEST_PATH = 'notebook/data/merged_test (1).csv'
MODEL_PATH = 'notebook/weights/lstm_model.pth'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

model = load_model(MODEL_PATH, input_size=train_df.shape[1] - 1)

preds = generate_predictions(model, train_df, test_df)

dates = test_df['date']

app = Dash(__name__)
app.layout = html.Div([
    html.H2('LSTM Close Price Prediction'),
    dcc.Graph(
        figure=go.Figure(
            data=[
                go.Scatter(x=dates, y=test_df['Close'], mode='lines', name='Actual'),
                go.Scatter(x=dates, y=preds, mode='lines', name='Predicted')
            ],
            layout=go.Layout(title='Actual vs Predicted Close')
        )
    )
])

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050)

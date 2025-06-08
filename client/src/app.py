import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go


def load_data():
    # Determine repository root to build dataset paths dynamically
    root = Path(__file__).resolve().parents[2]
    train_path = root / "notebook" / "data" / "merged_train (1).csv"
    test_path = root / "notebook" / "data" / "merged_test (1).csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train.columns = [c.lower().strip() for c in train.columns]
    test.columns = [c.lower().strip() for c in test.columns]
    train['date'] = pd.to_datetime(train['date'])
    test['date'] = pd.to_datetime(test['date'])
    data = pd.concat([train, test], axis=0).reset_index(drop=True)
    seq_length = 5
    pred_size = len(test)
    train = data[:-pred_size].copy()
    test = data[-(pred_size + seq_length):].copy()
    feature_cols = ['neu', 'trend', 'close']
    target_col = 'close'
    return train, test, feature_cols, target_col, seq_length


def prepare_sequences(train, test, feature_cols, target_col, seq_length):
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train[feature_cols])
    scaled_test = scaler.transform(test[feature_cols])

    close_scaler = MinMaxScaler()
    scaled_close_train = close_scaler.fit_transform(train[[target_col]]).flatten()
    scaled_close_test = close_scaler.transform(test[[target_col]]).flatten()

    def create_sequences(X, y, seq_length):
        xs, ys = [], []
        for i in range(len(X) - seq_length):
            xs.append(X[i:i + seq_length])
            ys.append(y[i + seq_length])
        return np.array(xs), np.array(ys)

    X_train, y_train = create_sequences(scaled_train, scaled_close_train, seq_length)
    X_test, y_test = create_sequences(scaled_test, scaled_close_test, seq_length)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train, y_train, X_test, y_test, close_scaler


class LSTM(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size=32, num_layers=2, output_size=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.seq_len, self.hidden_size),
            torch.zeros(self.num_layers, self.seq_len, self.hidden_size),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def train_model(X_train, y_train, seq_length, input_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM(input_size=input_size, seq_len=seq_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-5)
    criterion = nn.MSELoss()
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

    epochs = 30
    for epoch in range(1, epochs + 1):
        model.train()
        for x_data, y_data in train_loader:
            optimizer.zero_grad()
            model.reset_hidden_state()
            output = model(x_data.to(device))
            loss = criterion(output, y_data.to(device))
            loss.backward()
            optimizer.step()
    return model


def predict(model, X_test, close_scaler):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        model.reset_hidden_state()
        preds = model(X_test.to(device)).cpu().numpy().flatten()
    preds_inv = close_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    return preds_inv


def create_dash(df):
    app = dash.Dash(__name__)
    server = app.server
    app.layout = html.Div([
        html.H1("LSTM 주가 예측 대시보드"),
        dcc.Slider(
            id='day-slider',
            min=0,
            max=len(df) - 1,
            value=0,
            marks={i: str(date.date()) for i, date in enumerate(df['date'])},
            step=None
        ),
        dcc.Graph(id='prediction-graph'),
        html.Div(id='stats-output')
    ])

    @app.callback(
        Output('prediction-graph', 'figure'),
        Output('stats-output', 'children'),
        Input('day-slider', 'value')
    )
    def update_graph(selected_day):
        selected_date = df['date'].iloc[selected_day]
        sub_df = df.iloc[:selected_day + 1]
        fig_price = go.Figure([
            go.Scatter(x=sub_df['date'], y=sub_df['close'], mode='lines+markers', name='실제 Close'),
            go.Scatter(x=sub_df['date'], y=sub_df['predicted_close'], mode='lines+markers', name='예측 Close')
        ])
        fig_price.update_layout(title='예측 vs 실제 (일별 업데이트)', xaxis_title='날짜', yaxis_title='가격')
        pos = df['pos'].iloc[selected_day]
        neu = df['neu'].iloc[selected_day]
        neg = df['neg'].iloc[selected_day]
        sentiment_fig = dcc.Graph(
            figure=go.Figure(
                data=[go.Bar(x=['긍정 (pos)', '중립 (neu)', '부정 (neg)'], y=[pos, neu, neg])],
                layout=go.Layout(title='감성 분석 결과', yaxis=dict(title='비율 (%)'), xaxis=dict(title='감성 종류'))
            )
        )
        stats = html.Div([
            html.P(f"선택 날짜: {selected_date.date()}"),
            html.P(f"실제 Close: {df['close'].iloc[selected_day]:.2f}"),
            html.P(f"예측 Close: {df['predicted_close'].iloc[selected_day]:.2f}"),
            sentiment_fig
        ])
        return fig_price, stats

    return app


def main():
    train, test, feature_cols, target_col, seq_length = load_data()
    X_train, y_train, X_test, y_test, close_scaler = prepare_sequences(train, test, feature_cols, target_col, seq_length)
    model = train_model(X_train, y_train, seq_length, input_size=len(feature_cols))
    preds = predict(model, X_test, close_scaler)
    df = test.iloc[seq_length:].copy()
    df['predicted_close'] = preds
    app = create_dash(df)
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
